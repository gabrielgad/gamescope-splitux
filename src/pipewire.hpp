#pragma once

#include <atomic>
#include <memory>
#include <pipewire/pipewire.h>
#include <pipewire/thread-loop.h>
#include <spa/param/video/format-utils.h>

#include "rendervulkan.hpp"
#include "pipewire_gamescope.hpp"

struct pipewire_state {
	struct pw_thread_loop *thread_loop;
	struct pw_context *context;
	struct pw_core *core;
	bool running;

	struct pw_stream *stream;
	std::atomic<uint32_t> stream_node_id;
	std::atomic<bool> streaming;
	struct spa_video_info_raw video_info;
	struct spa_gamescope gamescope_info;
	uint64_t focus_appid;
	bool dmabuf;
	int shm_stride;
	uint64_t seq;
};

/**
 * PipeWire buffers are allocated by the PipeWire thread, and are temporarily
 * lent to the steamcompmgr thread (via pipewire_dequeue_buffer and
 * pipewire_submit_buffer) for rendering+copying. Every access to the pipewire
 * buffer pool — dequeue, queue, trigger, add_buffer, remove_buffer — happens
 * under the thread-loop lock, so the two threads can never touch the pool
 * concurrently.
 */
struct pipewire_buffer {
	enum spa_data_type type; // SPA_DATA_MemFd or SPA_DATA_DmaBuf
	struct spa_video_info_raw video_info;
	struct spa_gamescope gamescope_info;
	gamescope::OwningRc<CVulkanTexture> texture;

	// Internal-only RGB->NV12 compute-shader target when `texture` is a
	// dmabuf-exported, tiled-modifier image (see stream_handle_add_buffer):
	// RADV rejects every exportable NV12 modifier -- including LINEAR-as-a-
	// modifier -- the instant VK_IMAGE_USAGE_STORAGE_BIT is requested, so the
	// compute shader can't write into the export texture directly on this
	// hardware. Null for the memfd path, where `texture` itself is still the
	// compute target (no modifier/export conflict there).
	gamescope::OwningRc<CVulkanTexture> compute_texture;

	// Only used for SPA_DATA_MemFd
	struct {
		int stride;
		uint8_t *data;
		int fd;
	} shm;

	// The PipeWire buffer, or nullptr if remove_buffer has detached it. Only
	// read/written under the thread-loop lock.
	struct pw_buffer *buffer;

	// True while the steamcompmgr (producer) thread holds this buffer for
	// render+copy. Only read/written under the thread-loop lock. The producer
	// runs its GPU work with the lock released, so remove_buffer consults this
	// flag to decide free-now vs defer-to-producer.
	bool in_producer;
};

bool init_pipewire(void);
void deinit_pipewire(void);
uint32_t get_pipewire_stream_node_id(void);
struct pipewire_buffer *pipewire_dequeue_buffer(void);
void pipewire_submit_buffer(struct pipewire_buffer *buffer);
bool pipewire_is_streaming();
bool pipewire_has_consumer();
void pipewire_destroy_buffer(struct pipewire_buffer *buffer);
