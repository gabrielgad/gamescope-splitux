
#include <assert.h>
#include <fcntl.h>
#include <poll.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <vector>

#include "main.hpp"
#include "pipewire.hpp"
#include "log.hpp"

#include <spa/debug/format.h>

static LogScope pwr_log("pipewire");

static struct pipewire_state pipewire_state = { .stream_node_id = SPA_ID_INVALID };

// Number of buffers allocated for a connected consumer. >0 means a consumer has
// linked and negotiated buffers; used to run the compositor capture while the
// stream is still PAUSED so it can bootstrap to STREAMING. Mutated only from the
// loop thread (add_buffer/remove_buffer), read from the steamcompmgr thread.
static std::atomic<int> s_nConsumerBuffers{0};

// Requested capture size
static uint32_t s_nRequestedWidth;
static uint32_t s_nRequestedHeight;
static uint32_t s_nCaptureWidth;
static uint32_t s_nCaptureHeight;
static uint32_t s_nOutputWidth;
static uint32_t s_nOutputHeight;

static void destroy_buffer(struct pipewire_buffer *buffer) {
	// The ownership protocol (every pool call under the thread-loop lock +
	// in_producer) now guarantees a buffer is freed exactly once, with its
	// pw_buffer already detached — so neither guard below should ever fire.
	// Kept as belt-and-suspenders: degrade a never-expected residual race to a
	// logged leak instead of a SIGABRT that kills the compositor mid-session.
	if (buffer->buffer != nullptr) {
		pwr_log.errorf("destroy_buffer: pw_buffer still attached (race?) — leaking buffer");
		return;
	}

	switch (buffer->type) {
	case SPA_DATA_MemFd:
	{
		off_t size = buffer->shm.stride * buffer->video_info.size.height;
		if (buffer->video_info.format == SPA_VIDEO_FORMAT_NV12) {
			size += buffer->shm.stride * ((buffer->video_info.size.height + 1) / 2);
		}
		munmap(buffer->shm.data, size);
		close(buffer->shm.fd);
		break;
	}
	case SPA_DATA_DmaBuf:
		break; // nothing to do
	default:
		pwr_log.errorf("destroy_buffer: unexpected buffer type %d (use-after-free?) — leaking buffer", (int)buffer->type);
		return;
	}

	delete buffer;
}

void pipewire_destroy_buffer(struct pipewire_buffer *buffer)
{
	destroy_buffer(buffer);
}

static void calculate_capture_size()
{
	s_nCaptureWidth = s_nOutputWidth;
	s_nCaptureHeight = s_nOutputHeight;

	if (s_nRequestedWidth > 0 && s_nRequestedHeight > 0 &&
	    (s_nOutputWidth > s_nRequestedWidth || s_nOutputHeight > s_nRequestedHeight)) {
		// Need to clamp to the smallest dimension
		float flRatioW = static_cast<float>(s_nRequestedWidth) / s_nOutputWidth;
		float flRatioH = static_cast<float>(s_nRequestedHeight) / s_nOutputHeight;
		if (flRatioW <= flRatioH) {
			s_nCaptureWidth = s_nRequestedWidth;
			s_nCaptureHeight = static_cast<uint32_t>(ceilf(flRatioW * s_nOutputHeight));
		} else {
			s_nCaptureWidth = static_cast<uint32_t>(ceilf(flRatioH * s_nOutputWidth));
			s_nCaptureHeight = s_nRequestedHeight;
		}
	}
}

static void build_format_params(struct spa_pod_builder *builder, spa_video_format format, std::vector<const struct spa_pod *> &params) {
	struct spa_rectangle size = SPA_RECTANGLE(s_nCaptureWidth, s_nCaptureHeight);
	struct spa_rectangle min_requested_size = { 0, 0 };
	struct spa_rectangle max_requested_size = { UINT32_MAX, UINT32_MAX };
	struct spa_fraction framerate = SPA_FRACTION(0, 1);

	// NV12 export modifiers: queried straight from the physical device, not
	// GetBackend()->UsesModifiers() (this image is exported to PipeWire and
	// never re-imported into the backend, so the parent compositor's scanout
	// modifier knowledge doesn't apply). Re-enabled 2026-07-01: live testing
	// on 2026-06-30 found the seat-streamer consumer's tiled import faulting
	// the GPU ("radv: GPUVM fault detected") because gst-dmabuf-vulkan's
	// meta-repair heuristic was unconditionally overwriting plane 1's real,
	// Vulkan-queried tiled offset/stride with a LINEAR-only packing formula;
	// fixed consumer-side (gstdmabufvulkan.c, gated that heuristic to
	// DRM_FORMAT_MOD_LINEAR only) — safe to offer tiled modifiers again.
	//
	// Usage is TRANSFER_DST only — deliberately NOT VK_IMAGE_USAGE_STORAGE_BIT.
	// This must match the actual export texture's usage in
	// stream_handle_add_buffer: RADV rejects every exportable NV12 modifier,
	// including LINEAR-as-a-modifier, the instant STORAGE_BIT is requested
	// (verified via a standalone probe), so querying with STORAGE_BIT here
	// always returned zero candidates and this function silently offered only
	// LINEAR regardless of what the driver can actually do.
	std::vector<uint64_t> modifiers = vulkan_get_exportable_modifiers(
		DRM_FORMAT_NV12, VK_IMAGE_USAGE_TRANSFER_DST_BIT);
	bool hasLinear = false;
	for (uint64_t m : modifiers) {
		if (m == DRM_FORMAT_MOD_LINEAR) {
			hasLinear = true;
			break;
		}
	}
	if (!hasLinear)
		modifiers.push_back(DRM_FORMAT_MOD_LINEAR);

	{
		std::string modlist;
		for (uint64_t m : modifiers) {
			char buf[32];
			snprintf(buf, sizeof(buf), "0x%llx ", (unsigned long long)m);
			modlist += buf;
		}
		pwr_log.infof("splitux: offering %zu NV12 export modifier candidate(s): %s", modifiers.size(), modlist.c_str());
	}

	struct spa_pod_frame obj_frame, choice_frame;
	spa_pod_builder_push_object(builder, &obj_frame, SPA_TYPE_OBJECT_Format, SPA_PARAM_EnumFormat);
	spa_pod_builder_add(builder,
		SPA_FORMAT_mediaType, SPA_POD_Id(SPA_MEDIA_TYPE_video),
		SPA_FORMAT_mediaSubtype, SPA_POD_Id(SPA_MEDIA_SUBTYPE_raw),
		SPA_FORMAT_VIDEO_format, SPA_POD_Id(format),
		SPA_FORMAT_VIDEO_size, SPA_POD_Rectangle(&size),
		SPA_FORMAT_VIDEO_framerate, SPA_POD_Fraction(&framerate),
		SPA_FORMAT_VIDEO_requested_size, SPA_POD_CHOICE_RANGE_Rectangle( &min_requested_size, &min_requested_size, &max_requested_size ),
		SPA_FORMAT_VIDEO_gamescope_focus_appid, SPA_POD_CHOICE_RANGE_Long( 0ll, INT64_MIN, INT64_MAX ),
		0);
	if (format == SPA_VIDEO_FORMAT_NV12) {
		spa_pod_builder_add(builder,
			SPA_FORMAT_VIDEO_colorMatrix, SPA_POD_CHOICE_ENUM_Id(3,
							SPA_VIDEO_COLOR_MATRIX_BT601,
							SPA_VIDEO_COLOR_MATRIX_BT601,
							SPA_VIDEO_COLOR_MATRIX_BT709),
			SPA_FORMAT_VIDEO_colorRange, SPA_POD_CHOICE_ENUM_Id(3,
							SPA_VIDEO_COLOR_RANGE_16_235,
							SPA_VIDEO_COLOR_RANGE_16_235,
							SPA_VIDEO_COLOR_RANGE_0_255),
			0);
	}
	// CORRECTED 2026-07-01 (same day, later in the session): an earlier A/B
	// test here concluded the consumer "genuinely cannot import ANY tiled
	// modifier" after pinning to a fixed tiled value made negotiation fail
	// outright. That test was confounded — at the time, the consumer's own
	// sink caps (gstdmabufvulkan.c) built a bare "NV12" drm-format string,
	// which GStreamer's own gst_video_dma_drm_fourcc_from_string() resolves
	// to DRM_FORMAT_MOD_LINEAR (there is no "let PipeWire negotiate it below
	// us" wildcard), so the consumer was only ever asking for LINEAR
	// regardless of what gamescope offered. Fixed consumer-side (explicit
	// per-modifier drm-format entries + a pipewiresrc default-choice
	// ordering fix) — re-ran the SAME single-fixed-modifier test afterward
	// and it succeeded (real tiled modifier negotiated + imported, no
	// faults). Real tiled import works; keep offering the full candidate
	// list (tiled preferred, LINEAR always last) as the safety net for a
	// driver/consumer that genuinely can't do better.
	spa_pod_builder_prop(builder, SPA_FORMAT_VIDEO_modifier, SPA_POD_PROP_FLAG_MANDATORY);
	spa_pod_builder_push_choice(builder, &choice_frame, SPA_CHOICE_Enum, 0);
	spa_pod_builder_long(builder, modifiers[0]); // default: prefer the backend's first (tiled) modifier
	for (uint64_t m : modifiers)
		spa_pod_builder_long(builder, m);
	spa_pod_builder_pop(builder, &choice_frame);
	params.push_back((const struct spa_pod *) spa_pod_builder_pop(builder, &obj_frame));

	spa_pod_builder_push_object(builder, &obj_frame, SPA_TYPE_OBJECT_Format, SPA_PARAM_EnumFormat);
	spa_pod_builder_add(builder,
		SPA_FORMAT_mediaType, SPA_POD_Id(SPA_MEDIA_TYPE_video),
		SPA_FORMAT_mediaSubtype, SPA_POD_Id(SPA_MEDIA_SUBTYPE_raw),
		SPA_FORMAT_VIDEO_format, SPA_POD_Id(format),
		SPA_FORMAT_VIDEO_size, SPA_POD_Rectangle(&size),
		SPA_FORMAT_VIDEO_framerate, SPA_POD_Fraction(&framerate),
		SPA_FORMAT_VIDEO_requested_size, SPA_POD_CHOICE_RANGE_Rectangle( &min_requested_size, &min_requested_size, &max_requested_size ),
		SPA_FORMAT_VIDEO_gamescope_focus_appid, SPA_POD_CHOICE_RANGE_Long( 0ll, INT64_MIN, INT64_MAX ),
		0);
	if (format == SPA_VIDEO_FORMAT_NV12) {
		spa_pod_builder_add(builder,
			SPA_FORMAT_VIDEO_colorMatrix, SPA_POD_CHOICE_ENUM_Id(3,
							SPA_VIDEO_COLOR_MATRIX_BT601,
							SPA_VIDEO_COLOR_MATRIX_BT601,
							SPA_VIDEO_COLOR_MATRIX_BT709),
			SPA_FORMAT_VIDEO_colorRange, SPA_POD_CHOICE_ENUM_Id(3,
							SPA_VIDEO_COLOR_RANGE_16_235,
							SPA_VIDEO_COLOR_RANGE_16_235,
							SPA_VIDEO_COLOR_RANGE_0_255),
			0);
	}
	params.push_back((const struct spa_pod *) spa_pod_builder_pop(builder, &obj_frame));

//	for (auto& param : params)
//		spa_debug_format(2, nullptr, param);
}


static std::vector<const struct spa_pod *> build_format_params(struct spa_pod_builder *builder)
{
	std::vector<const struct spa_pod *> params;

	build_format_params(builder, SPA_VIDEO_FORMAT_BGRx, params);
	build_format_params(builder, SPA_VIDEO_FORMAT_NV12, params);

	return params;
}

// Fold in any output-size change that happened since the last frame and, if the
// resulting capture size differs from the negotiated format, renegotiate. Must
// be called with the thread-loop lock held (it touches the stream params).
static void maybe_renegotiate_size_locked(struct pipewire_state *state)
{
	if (g_nOutputWidth != s_nOutputWidth || g_nOutputHeight != s_nOutputHeight) {
		s_nOutputWidth = g_nOutputWidth;
		s_nOutputHeight = g_nOutputHeight;
		calculate_capture_size();
	}
	if (s_nCaptureWidth != state->video_info.size.width || s_nCaptureHeight != state->video_info.size.height) {
		pwr_log.debugf("renegotiating stream params (size: %dx%d)", s_nCaptureWidth, s_nCaptureHeight);

		uint8_t buf[4096];
		struct spa_pod_builder builder = SPA_POD_BUILDER_INIT(buf, sizeof(buf));
		std::vector<const struct spa_pod *> format_params = build_format_params(&builder);
		int ret = pw_stream_update_params(state->stream, format_params.data(), format_params.size());
		if (ret < 0) {
			pwr_log.errorf("pw_stream_update_params failed");
		}
	}
}

static void copy_buffer(struct pipewire_state *state, struct pipewire_buffer *buffer)
{
	gamescope::OwningRc<CVulkanTexture> &tex = buffer->texture;
	assert(tex != nullptr);

	struct pw_buffer *pw_buffer = buffer->buffer;
	struct spa_buffer *spa_buffer = pw_buffer->buffer;

	bool needs_reneg = buffer->video_info.size.width != tex->width() || buffer->video_info.size.height != tex->height();

	struct spa_meta_header *header = (struct spa_meta_header *) spa_buffer_find_meta_data(spa_buffer, SPA_META_Header, sizeof(*header));
	if (header != nullptr) {
		header->pts = -1;
		header->flags = needs_reneg ? SPA_META_HEADER_FLAG_CORRUPTED : 0;
		header->seq = state->seq++;
		header->dts_offset = 0;
	}

	float *requested_size_scale = (float *) spa_buffer_find_meta_data(spa_buffer, SPA_META_requested_size_scale, sizeof(*requested_size_scale));
	if (requested_size_scale != nullptr) {
		*requested_size_scale = ((float)tex->width() / g_nOutputWidth);
	}

	struct wlr_dmabuf_attributes dmabuf;
	switch (buffer->type) {
	case SPA_DATA_MemFd: {
		struct spa_chunk *chunk = spa_buffer->datas[0].chunk;
		chunk->flags = needs_reneg ? SPA_CHUNK_FLAG_CORRUPTED : 0;
		chunk->offset = 0;
		chunk->size = state->video_info.size.height * buffer->shm.stride;
		if (state->video_info.format == SPA_VIDEO_FORMAT_NV12) {
			chunk->size += ((state->video_info.size.height + 1)/2 * buffer->shm.stride);
		}
		chunk->stride = buffer->shm.stride;

		if (!needs_reneg) {
			uint8_t *pMappedData = tex->mappedData();

			if (state->video_info.format == SPA_VIDEO_FORMAT_NV12) {
				for (uint32_t i = 0; i < tex->height(); i++) {
					const uint32_t lumaPwOffset = 0;
					memcpy(
						&buffer->shm.data[lumaPwOffset      + i * buffer->shm.stride],
						&pMappedData     [tex->lumaOffset() + i * tex->lumaRowPitch()],
						std::min<size_t>(buffer->shm.stride, tex->lumaRowPitch()));
				}

				for (uint32_t i = 0; i < (tex->height() + 1) / 2; i++) {
					const uint32_t chromaPwOffset = tex->height() * buffer->shm.stride;
					memcpy(
						&buffer->shm.data[chromaPwOffset      + i * buffer->shm.stride],
						&pMappedData     [tex->chromaOffset() + i * tex->chromaRowPitch()],
						std::min<size_t>(buffer->shm.stride, tex->chromaRowPitch()));
				}
			}
			else
			{
				for (uint32_t i = 0; i < tex->height(); i++) {
					memcpy(
						&buffer->shm.data[i * buffer->shm.stride],
						&pMappedData     [i * tex->rowPitch()],
						std::min<size_t>(buffer->shm.stride, tex->rowPitch()));
				}
			}
		}
		break;
	}
	case SPA_DATA_DmaBuf:
		dmabuf = tex->dmabuf();
		for (int i = 0; i < dmabuf.n_planes; i++) {
			struct spa_chunk *plane_chunk = spa_buffer->datas[i].chunk;
			plane_chunk->flags = needs_reneg ? SPA_CHUNK_FLAG_CORRUPTED : 0;
			plane_chunk->offset = dmabuf.offset[i];
			plane_chunk->stride = dmabuf.stride[i];
			if (state->video_info.format == SPA_VIDEO_FORMAT_NV12 && dmabuf.n_planes == 1) {
				// Single combined plane: luma then chroma packed sequentially
				// (matches the MemFd layout above).
				plane_chunk->size = dmabuf.height * plane_chunk->stride;
				plane_chunk->size += ((dmabuf.height + 1)/2 * plane_chunk->stride);
			} else if (i == 0) {
				plane_chunk->size = dmabuf.height * plane_chunk->stride;
			} else {
				// Separate chroma memory plane (4:2:0 subsampled height).
				plane_chunk->size = ((dmabuf.height + 1)/2) * plane_chunk->stride;
			}
		}
		break;
	default:
		assert(false); // unreachable
	}
}

static void stream_handle_state_changed(void *data, enum pw_stream_state old_stream_state, enum pw_stream_state stream_state, const char *error)
{
	struct pipewire_state *state = (struct pipewire_state *) data;

	pwr_log.infof("stream state changed: %s", pw_stream_state_as_string(stream_state));

	switch (stream_state) {
	case PW_STREAM_STATE_PAUSED:
		if (state->stream_node_id == SPA_ID_INVALID) {
			state->stream_node_id = pw_stream_get_node_id(state->stream);
			pwr_log.infof("stream available on node ID: %u", state->stream_node_id.load());
		}
		state->streaming = false;
		state->seq = 0;
		// Activate so the stream can progress to STREAMING; we then clock the
		// graph ourselves, one cycle per produced frame, from
		// pipewire_submit_buffer() via pw_stream_trigger_process().
		pw_stream_set_active(state->stream, true);
		break;
	case PW_STREAM_STATE_STREAMING:
		state->streaming = true;
		break;
	case PW_STREAM_STATE_ERROR:
	case PW_STREAM_STATE_UNCONNECTED:
		state->running = false;
		break;
	default:
		break;
	}
}

uint32_t spa_format_to_drm(uint32_t spa_format);

static void stream_handle_param_changed(void *data, uint32_t id, const struct spa_pod *param)
{
	struct pipewire_state *state = (struct pipewire_state *) data;

	if (param == nullptr || id != SPA_PARAM_Format)
		return;

	struct spa_gamescope gamescope_info{};

	int ret = spa_format_video_raw_parse_with_gamescope(param, &state->video_info, &gamescope_info);
	if (ret < 0) {
		pwr_log.errorf("spa_format_video_raw_parse failed");
		return;
	}
	s_nRequestedWidth = gamescope_info.requested_size.width;
	s_nRequestedHeight = gamescope_info.requested_size.height;
	calculate_capture_size();

	state->gamescope_info = gamescope_info;

	int bpp = 4;
	if (state->video_info.format == SPA_VIDEO_FORMAT_NV12) {
		bpp = 1;
	}

	state->shm_stride = SPA_ROUND_UP_N(state->video_info.size.width * bpp, 4);

	const struct spa_pod_prop *modifier_prop = spa_pod_find_prop(param, nullptr, SPA_FORMAT_VIDEO_modifier);
	state->dmabuf = modifier_prop != nullptr;

	uint8_t buf[1024];
	struct spa_pod_builder builder = SPA_POD_BUILDER_INIT(buf, sizeof(buf));

	// 48 (range max 64): in-flight buffer depth scales with capture fps. At the
	// 200fps tier the producer's render depth plus a zero-copy consumer
	// (splitux-together seat-streamer, va+always-copy) holding frames briefly
	// can otherwise starve capture ("out of buffers") and throttle the stream.
	// The SPA_POD_CHOICE_RANGE_Int max silently caps any larger request, so it
	// must be raised in lockstep. 48 dmabufs @1080p NV12 ≈ 160MB. min=1 lets
	// reneg shrink the pool.
	int buffers = 48;
	int shm_size = state->shm_stride * state->video_info.size.height;
	if (state->video_info.format == SPA_VIDEO_FORMAT_NV12) {
		shm_size += ((state->video_info.size.height + 1) / 2) * state->shm_stride;
	}
	int data_type = state->dmabuf ? (1 << SPA_DATA_DmaBuf) : (1 << SPA_DATA_MemFd);

	// A DMA-BUF modifier export (e.g. NV12 luma+chroma) needs one spa_data
	// "block" per real memory plane; reserve them up front since blocks isn't
	// renegotiable per-buffer. Only non-dmabuf (memfd/shm — no modifier at
	// all, DRM_FORMAT_MOD_INVALID) stays at 1: DRM_FORMAT_MOD_LINEAR used AS
	// A MODIFIER (the dmabuf-export path) still reports 2 real planes for
	// NV12 on this driver, same as any other modifier — see
	// vulkan_get_drm_format_modifier_plane_count.
	int blocks = 1;
	if (state->dmabuf) {
		uint32_t drmFormat = spa_format_to_drm(state->video_info.format);
		blocks = (int) vulkan_get_drm_format_modifier_plane_count(drmFormat, state->video_info.modifier);
	}
	pwr_log.infof("splitux: negotiated modifier=0x%llx blocks=%d dmabuf=%d",
		(unsigned long long) state->video_info.modifier, blocks, (int) state->dmabuf);

	const struct spa_pod *buffers_param =
		(const struct spa_pod *) spa_pod_builder_add_object(&builder,
		SPA_TYPE_OBJECT_ParamBuffers, SPA_PARAM_Buffers,
		SPA_PARAM_BUFFERS_buffers, SPA_POD_CHOICE_RANGE_Int(buffers, 1, 64),
		SPA_PARAM_BUFFERS_blocks, SPA_POD_Int(blocks),
		SPA_PARAM_BUFFERS_size, SPA_POD_Int(shm_size),
		SPA_PARAM_BUFFERS_stride, SPA_POD_Int(state->shm_stride),
		SPA_PARAM_BUFFERS_dataType, SPA_POD_CHOICE_FLAGS_Int(data_type));
	const struct spa_pod *meta_param =
		(const struct spa_pod *) spa_pod_builder_add_object(&builder,
		SPA_TYPE_OBJECT_ParamMeta, SPA_PARAM_Meta,
		SPA_PARAM_META_type, SPA_POD_Id(SPA_META_Header),
		SPA_PARAM_META_size, SPA_POD_Int(sizeof(struct spa_meta_header)));
	const struct spa_pod *scale_param =
		(const struct spa_pod *) spa_pod_builder_add_object(&builder,
		SPA_TYPE_OBJECT_ParamMeta, SPA_PARAM_Meta,
		SPA_PARAM_META_type, SPA_POD_Id(SPA_META_requested_size_scale),
		SPA_PARAM_META_size, SPA_POD_Int(sizeof(float)));
	const struct spa_pod *params[] = { buffers_param, meta_param, scale_param };

	ret = pw_stream_update_params(state->stream, params, sizeof(params) / sizeof(params[0]));
	if (ret != 0) {
		pwr_log.errorf("pw_stream_update_params failed");
	}

	pwr_log.debugf("format changed (size: %dx%d, requested %dx%d, format %d, stride %d, size: %d, dmabuf: %d)",
		state->video_info.size.width, state->video_info.size.height,
		s_nRequestedWidth, s_nRequestedHeight,
		state->video_info.format, state->shm_stride, shm_size, state->dmabuf);
}

static void randname(char *buf)
{
	struct timespec ts;
	clock_gettime(CLOCK_REALTIME, &ts);
	long r = ts.tv_nsec;
	for (int i = 0; i < 6; ++i) {
		buf[i] = 'A'+(r&15)+(r&16)*2;
		r >>= 5;
	}
}

static int anonymous_shm_open(void)
{
	char name[] = "/gamescope-pw-XXXXXX";
	int retries = 100;

	do {
		randname(name + strlen(name) - 6);

		--retries;
		// shm_open guarantees that O_CLOEXEC is set
		int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0600);
		if (fd >= 0) {
			shm_unlink(name);
			return fd;
		}
	} while (retries > 0 && errno == EEXIST);

	return -1;
}

uint32_t spa_format_to_drm(uint32_t spa_format)
{
	switch (spa_format)
	{
		case SPA_VIDEO_FORMAT_NV12: return DRM_FORMAT_NV12;
		default:
		case SPA_VIDEO_FORMAT_BGR: return DRM_FORMAT_XRGB8888;
	}
}

static void stream_handle_add_buffer(void *user_data, struct pw_buffer *pw_buffer)
{
	struct pipewire_state *state = (struct pipewire_state *) user_data;

	struct spa_buffer *spa_buffer = pw_buffer->buffer;
	struct spa_data *spa_data = &spa_buffer->datas[0];

	struct pipewire_buffer *buffer = new pipewire_buffer();
	buffer->buffer = pw_buffer;
	buffer->video_info = state->video_info;
	buffer->gamescope_info = state->gamescope_info;

	bool is_dmabuf = (spa_data->type & (1 << SPA_DATA_DmaBuf)) != 0;
	bool is_memfd = (spa_data->type & (1 << SPA_DATA_MemFd)) != 0;

	EStreamColorspace colorspace = k_EStreamColorspace_Unknown;
	switch (state->video_info.color_matrix) {
	case SPA_VIDEO_COLOR_MATRIX_BT601:
		switch (state->video_info.color_range) {
		case SPA_VIDEO_COLOR_RANGE_16_235:
			colorspace = k_EStreamColorspace_BT601;
			break;
		case SPA_VIDEO_COLOR_RANGE_0_255:
			colorspace = k_EStreamColorspace_BT601_Full;
			break;
		default:
			break;
		}
		break;
	case SPA_VIDEO_COLOR_MATRIX_BT709:
		switch (state->video_info.color_range) {
		case SPA_VIDEO_COLOR_RANGE_16_235:
			colorspace = k_EStreamColorspace_BT709;
			break;
		case SPA_VIDEO_COLOR_RANGE_0_255:
			colorspace = k_EStreamColorspace_BT709_Full;
			break;
		default:
			break;
		}
		break;
	default:
		break;
	}

	uint32_t drmFormat = spa_format_to_drm(state->video_info.format);

	buffer->texture = new CVulkanTexture();
	CVulkanTexture::createFlags screenshotImageFlags;
	screenshotImageFlags.bTransferDst = true;
	if (is_dmabuf)
	{
		// Export natively tiled (the encoder's preferred layout) instead of
		// forcing LINEAR: avoids the linear->tiled relayout the consumer would
		// otherwise have to do on the GPU (contending with game render). Not
		// bMappable — dmabuf consumers import the fd, they don't CPU-map it.
		//
		// Deliberately NOT bStorage here. Verified via a standalone Vulkan
		// probe against this GPU/driver: RADV exposes real tiled NV12
		// modifiers (8 of them on RX 9070/GFX1201), all fully exportable --
		// but every one, including plain LINEAR-as-a-modifier, loses
		// EXTERNAL_MEMORY_FEATURE_EXPORTABLE the instant
		// VK_IMAGE_USAGE_STORAGE_BIT is in the usage set. Requesting it here
		// unconditionally forced the "no exportable tiled modifier" LINEAR
		// fallback below, every frame, regardless of what the driver can
		// actually do. The RGB->NV12 compute shader (vulkan_screenshot) still
		// needs a STORAGE-capable write target, so that's `compute_texture`
		// now (OPTIMAL, internal-only, allocated below) — paint_pipewire
		// copies its output into this export texture afterward
		// (CVulkanCmdBuffer::copyImage, chained into the same command buffer
		// as the compute dispatch), which only needs TRANSFER_DST/_SRC.
		screenshotImageFlags.bExportable = true;
		screenshotImageFlags.bExportTiled = true;
		// Pin to the modifier already fixated by SPA format negotiation (the
		// consumer was already told the resulting plane count via this same
		// state->video_info.modifier -> blocks in stream_handle_param_changed)
		// so the texture BInit actually creates can't diverge from it.
		if (state->video_info.modifier != DRM_FORMAT_MOD_INVALID)
			screenshotImageFlags.uExplicitExportModifier = state->video_info.modifier;
	}
	else
	{
		// memfd path: the compute shader writes directly into this texture
		// (no separate export step, so no modifier/STORAGE conflict — this
		// isn't modifier-tiled at all), so it keeps bStorage.
		screenshotImageFlags.bStorage = true;
		screenshotImageFlags.bMappable = true;
		if (drmFormat == DRM_FORMAT_NV12)
		{
			screenshotImageFlags.bExportable = true;
			screenshotImageFlags.bLinear = true;
		}
	}
	bool bImageInitSuccess = buffer->texture->BInit( s_nCaptureWidth, s_nCaptureHeight, 1u, drmFormat, screenshotImageFlags );
	if ( !bImageInitSuccess )
	{
		pwr_log.errorf("Failed to initialize pipewire texture");
		goto error;
	}
	buffer->texture->setStreamColorspace(colorspace);

	if (is_dmabuf && drmFormat == DRM_FORMAT_NV12)
	{
		// See the flags comment above: the export texture above can't be the
		// compute shader's write target on this hardware, so give it a
		// separate OPTIMAL-tiled, STORAGE-capable scratch target instead. It
		// never leaves the GPU (no bExportable/bExportTiled/bLinear/bMappable),
		// so its own tiling can be whatever RADV prefers for fastest
		// storage-image writes.
		buffer->compute_texture = new CVulkanTexture();
		CVulkanTexture::createFlags computeImageFlags;
		computeImageFlags.bStorage = true;
		computeImageFlags.bTransferSrc = true;
		if ( !buffer->compute_texture->BInit( s_nCaptureWidth, s_nCaptureHeight, 1u, drmFormat, computeImageFlags ) )
		{
			pwr_log.errorf("Failed to initialize pipewire compute texture");
			goto error;
		}
		buffer->compute_texture->setStreamColorspace(colorspace);
	}

	if (is_dmabuf) {
		const struct wlr_dmabuf_attributes dmabuf = buffer->texture->dmabuf();
		if (dmabuf.n_planes < 1 || (size_t) dmabuf.n_planes > SPA_N_ELEMENTS(dmabuf.fd))
		{
			pwr_log.errorf("dmabuf.n_planes out of range (%d)", dmabuf.n_planes);
			goto error;
		}
		if ((uint32_t) dmabuf.n_planes > spa_buffer->n_datas)
		{
			// blocks negotiated via SPA_PARAM_Buffers (stream_handle_param_changed)
			// didn't reserve enough data slots for this modifier's plane count.
			pwr_log.errorf("dmabuf.n_planes (%d) > spa_buffer->n_datas (%d)", dmabuf.n_planes, spa_buffer->n_datas);
			goto error;
		}

		// A tiled multi-plane export (e.g. NV12 luma+chroma) may share a single
		// underlying dmabuf across planes (one fd dup'd per plane, see
		// CVulkanTexture::BInit), so every dup'd fd reports the same overall size.
		off_t size = lseek(dmabuf.fd[0], 0, SEEK_END);
		if (size < 0) {
			pwr_log.errorf_errno("lseek failed");
			goto error;
		}

		buffer->type = SPA_DATA_DmaBuf;

		for (int i = 0; i < dmabuf.n_planes; i++)
		{
			struct spa_data *plane_data = &spa_buffer->datas[i];
			plane_data->type = SPA_DATA_DmaBuf;
			plane_data->flags = SPA_DATA_FLAG_READABLE;
			plane_data->fd = dmabuf.fd[i];
			plane_data->mapoffset = dmabuf.offset[i];
			plane_data->maxsize = size;
			plane_data->data = nullptr;
		}
	} else if (is_memfd) {
		int fd = anonymous_shm_open();
		if (fd < 0) {
			pwr_log.errorf("failed to create shm file");
			goto error;
		}

		off_t size = state->shm_stride * state->video_info.size.height;
		if (state->video_info.format == SPA_VIDEO_FORMAT_NV12) {
			size += state->shm_stride * ((state->video_info.size.height + 1) / 2);
		}
		if (ftruncate(fd, size) != 0) {
			pwr_log.errorf_errno("ftruncate failed");
			close(fd);
			goto error;
		}

		void *data = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
		if (data == MAP_FAILED) {
			pwr_log.errorf_errno("mmap failed");
			close(fd);
			goto error;
		}

		buffer->type = SPA_DATA_MemFd;
		buffer->shm.stride = state->shm_stride;
		buffer->shm.data = (uint8_t *) data;
		buffer->shm.fd = fd;

		spa_data->type = SPA_DATA_MemFd;
		spa_data->flags = SPA_DATA_FLAG_READABLE;
		spa_data->fd = fd;
		spa_data->mapoffset = 0;
		spa_data->maxsize = size;
		spa_data->data = data;
	} else {
		pwr_log.errorf("unsupported data type");
		spa_data->type = SPA_DATA_Invalid;
		goto error;
	}

	pw_buffer->user_data = buffer;

	s_nConsumerBuffers.fetch_add(1, std::memory_order_relaxed);

	return;

error:
	delete buffer;
}

static void stream_handle_remove_buffer(void *data, struct pw_buffer *pw_buffer)
{
	struct pipewire_buffer *buffer = (struct pipewire_buffer *) pw_buffer->user_data;

	s_nConsumerBuffers.fetch_sub(1, std::memory_order_relaxed);

	// Runs on the loop thread with the thread-loop lock effectively held, so it
	// is serialized against pipewire_dequeue_buffer/pipewire_submit_buffer.
	// Detach the pw_buffer. If the producer is mid-render with it
	// (in_producer == true), leave the free to its submit; otherwise we are the
	// sole owner and free now. destroy_buffer runs the CVulkanTexture dtor, but
	// no GPU work is in flight for a buffer the producer isn't holding.
	buffer->buffer = nullptr;
	if (!buffer->in_producer) {
		destroy_buffer(buffer);
	}
}

// We are a DRIVER node, so we must schedule the graph's cycles ourselves. With
// the thread-loop design this is demand-driven: pipewire_submit_buffer()
// triggers exactly one cycle per produced frame, so delivery rate == production
// rate with no fixed cadence. .process must exist for the driver but the
// producing happens on the steamcompmgr thread, so it is a no-op here.
static void stream_handle_process(void *data)
{
	(void) data;
}

static const struct pw_stream_events stream_events = {
	.version = PW_VERSION_STREAM_EVENTS,
	.state_changed = stream_handle_state_changed,
	.param_changed = stream_handle_param_changed,
	.add_buffer = stream_handle_add_buffer,
	.remove_buffer = stream_handle_remove_buffer,
	.process = stream_handle_process,
};

bool init_pipewire(void)
{
	struct pipewire_state *state = &pipewire_state;

	pw_init(nullptr, nullptr);

	state->thread_loop = pw_thread_loop_new("gamescope-pw", nullptr);
	if (!state->thread_loop) {
		pwr_log.errorf("pw_thread_loop_new failed");
		return false;
	}

	// Build the whole object graph before starting the loop. While the loop
	// thread isn't spinning, no lock is needed.
	state->context = pw_context_new(pw_thread_loop_get_loop(state->thread_loop), nullptr, 0);
	if (!state->context) {
		pwr_log.errorf("pw_context_new failed");
		return false;
	}

	state->core = pw_context_connect(state->context, nullptr, 0);
	if (!state->core) {
		pwr_log.errorf("pw_context_connect failed");
		return false;
	}

	// Unique PipeWire node name per instance. With multiple gamescope instances
	// (splitux-together multi-seat) every node otherwise advertises the same
	// name, so a consumer matching by name (seat-streamer --pw-name) binds them
	// all to the FIRST node — every seat captures one instance and the rest are
	// orphaned. GAMESCOPE_PIPEWIRE_NODE lets the launcher give each instance a
	// distinct, targetable node name; default keeps the historical "gamescope".
	const char *pwNodeName = getenv("GAMESCOPE_PIPEWIRE_NODE");
	if (!pwNodeName || !*pwNodeName)
		pwNodeName = "gamescope";

	state->stream = pw_stream_new(state->core, pwNodeName,
		pw_properties_new(
			PW_KEY_MEDIA_CLASS, "Video/Source",
			PW_KEY_NODE_NAME, pwNodeName,
			nullptr));
	if (!state->stream) {
		pwr_log.errorf("pw_stream_new failed");
		return false;
	}

	static struct spa_hook stream_hook;
	pw_stream_add_listener(state->stream, &stream_hook, &stream_events, state);

	s_nRequestedWidth = 0;
	s_nRequestedHeight = 0;
	s_nOutputWidth = g_nOutputWidth;
	s_nOutputHeight = g_nOutputHeight;
	calculate_capture_size();

	uint8_t buf[4096];
	struct spa_pod_builder builder = SPA_POD_BUILDER_INIT(buf, sizeof(buf));
	std::vector<const struct spa_pod *> format_params = build_format_params(&builder);

	// Self-clocking driver: DRIVER so we own the graph clock (set_active on
	// PAUSED + a pw_stream_trigger_process() per produced frame), INACTIVE so
	// activation is explicit. This is what lets the capture run under a generic
	// session manager (WirePlumber), not only the Steam Deck's. The node id is
	// published asynchronously in state_changed on PAUSED.
	enum pw_stream_flags flags = (enum pw_stream_flags)(PW_STREAM_FLAG_DRIVER | PW_STREAM_FLAG_ALLOC_BUFFERS | PW_STREAM_FLAG_INACTIVE);
	int ret = pw_stream_connect(state->stream, PW_DIRECTION_OUTPUT, PW_ID_ANY, flags, format_params.data(), format_params.size());
	if (ret != 0) {
		pwr_log.errorf("pw_stream_connect failed");
		return false;
	}

	state->running = true;

	if (pw_thread_loop_start(state->thread_loop) < 0) {
		pwr_log.errorf("pw_thread_loop_start failed");
		return false;
	}

	return true;
}

void deinit_pipewire(void)
{
	struct pipewire_state *state = &pipewire_state;

	if (!state->thread_loop)
		return;

	// Must be called without the lock held.
	pw_thread_loop_stop(state->thread_loop);

	if (state->stream)
		pw_stream_destroy(state->stream);
	if (state->core)
		pw_core_disconnect(state->core);
	if (state->context)
		pw_context_destroy(state->context);
	pw_thread_loop_destroy(state->thread_loop);

	state->stream = nullptr;
	state->core = nullptr;
	state->context = nullptr;
	state->thread_loop = nullptr;
}

uint32_t get_pipewire_stream_node_id(void)
{
	return pipewire_state.stream_node_id.load();
}

bool pipewire_is_streaming()
{
	struct pipewire_state *state = &pipewire_state;
	return state->streaming;
}

bool pipewire_has_consumer()
{
	return s_nConsumerBuffers.load(std::memory_order_relaxed) > 0;
}

// steamcompmgr thread: lend a buffer to the producer for render+copy. The lock
// is held only around the cheap pool calls — never across the GPU work that
// follows in paint_pipewire — so the pw graph thread is not stalled for a frame.
struct pipewire_buffer *pipewire_dequeue_buffer(void)
{
	struct pipewire_state *state = &pipewire_state;

	// Produce while streaming, or while a consumer's buffers exist but the
	// stream is still PAUSED — the latter bootstraps the DRIVER stream to
	// STREAMING (its first queued frame completes the handshake).
	if (!(state->streaming || pipewire_has_consumer()))
		return nullptr;

	struct pipewire_buffer *buffer = nullptr;

	pw_thread_loop_lock(state->thread_loop);
	maybe_renegotiate_size_locked(state);
	struct pw_buffer *pw_buffer = pw_stream_dequeue_buffer(state->stream);
	// Consecutive dequeue failures. Transient starvation (producer briefly
	// outrunning the consumer) always recovers within a handful of frames via
	// the trigger below; hundreds of CONSECUTIVE failures only happen in the
	// permanently-wedged state (observed 2026-07-02: game video-mode changes
	// during world-load transitions leave every pool buffer stranded on the
	// consumer side forever — the stream then re-delivers one stale frame
	// eternally while all rate counters look alive).
	static uint32_t s_nConsecutiveDequeueFails = 0;
	if (pw_buffer) {
		s_nConsecutiveDequeueFails = 0;
		buffer = (struct pipewire_buffer *) pw_buffer->user_data;
		buffer->in_producer = true;
	} else {
		// Pool momentarily drained: above the consumer's recycle rate (e.g. a
		// 200fps producer vs a ~130fps encoder) the in-flight buffers can all be
		// queued/held at once. The graph is driven only from submit, so an empty
		// pool would mean no submit → no trigger → the consumer never runs its
		// cycle → it never releases what it holds → permanent deadlock ("out of
		// buffers" storm, cap=0). Drive a cycle here so the consumer drains its
		// backlog and recycles a buffer for the next frame. Rate-limit the log so
		// a transient drain doesn't flood it.
		if (pw_stream_is_driving(state->stream))
			pw_stream_trigger_process(state->stream);
		static int s_nOOB = 0;
		if ((s_nOOB++ % 200) == 0)
			pwr_log.errorf("warning: out of buffers (draining consumer backlog)");

		// SELF-HEAL for the permanent wedge: past the threshold, tear the
		// stream down and reconnect it in place — full param renegotiation,
		// fresh buffer pool, new node serial. The consumer (seat-streamer's
		// pipewiresrc, bound to the old serial) errors out and its session
		// rebuilds against the new node by NAME, and the browser auto-rejoins
		// — every link of that recovery chain is existing, exercised behavior.
		// Net effect: a ~2s video blip instead of a freeze-until-relaunch.
		// Threshold ≈ 3-10s of paint attempts with not one success; reset on
		// every successful dequeue so transient drains can never trip it.
		if (++s_nConsecutiveDequeueFails >= 600) {
			pwr_log.errorf("export pool starved for %u consecutive paints — reconnecting the stream to rebuild the buffer pool", s_nConsecutiveDequeueFails);
			s_nConsecutiveDequeueFails = 0;
			pw_stream_disconnect(state->stream);
			uint8_t buf[4096];
			struct spa_pod_builder builder = SPA_POD_BUILDER_INIT(buf, sizeof(buf));
			std::vector<const struct spa_pod *> format_params = build_format_params(&builder);
			enum pw_stream_flags flags = (enum pw_stream_flags)(PW_STREAM_FLAG_DRIVER | PW_STREAM_FLAG_ALLOC_BUFFERS | PW_STREAM_FLAG_INACTIVE);
			int ret = pw_stream_connect(state->stream, PW_DIRECTION_OUTPUT, PW_ID_ANY, flags, format_params.data(), format_params.size());
			if (ret != 0)
				pwr_log.errorf("self-heal pw_stream_connect failed (%d) — capture stays down until session relaunch", ret);
		}
	}
	pw_thread_loop_unlock(state->thread_loop);

	return buffer;
}

// steamcompmgr thread: hand a rendered buffer back to PipeWire. Called after the
// GPU render and vulkan_wait have completed with NO lock held. Takes the lock
// only for the cheap chunk/meta copy + queue + trigger.
void pipewire_submit_buffer(struct pipewire_buffer *buffer)
{
	struct pipewire_state *state = &pipewire_state;

	pw_thread_loop_lock(state->thread_loop);

	buffer->in_producer = false;

	if (buffer->buffer == nullptr) {
		// remove_buffer fired while we were rendering and deferred the free to
		// us. We're the sole owner now; free once, outside the lock.
		pw_thread_loop_unlock(state->thread_loop);
		destroy_buffer(buffer);
		return;
	}

	copy_buffer(state, buffer);

	int ret = pw_stream_queue_buffer(state->stream, buffer->buffer);
	if (ret < 0) {
		pwr_log.errorf("pw_stream_queue_buffer failed");
	}

	// Demand-driven drive: one cycle per produced frame. Only a driving node
	// (STREAMING + driver) may trigger; PAUSED→STREAMING still bootstraps via
	// the queued frame above. The trigger re-routes the cycle onto the data
	// loop via a non-blocking pw_loop_invoke, so it is safe under the lock.
	if (pw_stream_is_driving(state->stream))
		pw_stream_trigger_process(state->stream);

	pw_thread_loop_unlock(state->thread_loop);
}
