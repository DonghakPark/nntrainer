// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file   fsu_weight_pool.h
 * @date   10 July 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  FSU Weight Pool
 *
 */

#include <fsu_weight_pool.h>

namespace nntrainer {

FsuWeightPool::FsuWeightPool() : fd(-1), load_batch_size(1) {
  // load_task_executor = new TaskExecutor("loadPool", 8);
}

FsuWeightPool::~FsuWeightPool() {
  try {
    FsuWeightPool::deallocate();
  } catch (...) {
    ml_loge("Failed deallocate");
  }
  if (load_task_executor) {
    delete load_task_executor;
    load_task_executor = nullptr;
  }
}

void FsuWeightPool::weightFileOpen() {
  if (fd > 0)
    return;
  fd = open(weight_file_path.c_str(), O_RDWR | O_CREAT, 0666UL);
  NNTR_THROW_IF(fd < 0, std::runtime_error)
    << "[FSU_ELEM] Open file Failed : " << weight_file_path;
}

void FsuWeightPool::weightFileClose() {
  if (fd < 0) {
    return;
  }
  close(fd);
  fd = -1;
}

void FsuWeightPool::setWeightOffset(
  std::vector<std::pair<size_t, size_t>> offsets) {
  int id_idx = 1;
  for (auto element : offsets) {
    elements[id_idx].start_offset = element.first;
    elements[id_idx].weight_len = element.second;
    id_idx++;
  }
}

void FsuWeightPool::allocate() {

  size_t pool_size = size();
  NNTR_THROW_IF(pool_size == 0, std::runtime_error)
    << "Allocating memory pool with size 0";

  MemoryPool::allocateFSU();
}

void FsuWeightPool::deallocate() { MemoryPool::deallocate(); }

unsigned int FsuWeightPool::requestMemory(size_t bytes, unsigned int start_time,
                                          unsigned int end_time,
                                          std::vector<unsigned int> exec_order,
                                          TensorLifespan lifespan,
                                          bool is_wgrad) {
  auto id = MemoryPool::requestMemory(bytes, start_time, end_time, exec_order,
                                      lifespan, is_wgrad);
  return id;
}

std::shared_ptr<MemoryData> FsuWeightPool::getMemory(unsigned int id) {

  auto exe_order = getMemoryExecOrder().at(id - 1);

  void *memory_ptr = nullptr;
  memory_ptr = getMemoryPtrs().at(id - 1);

  auto mem_data = std::make_shared<MemoryData>(
    id, std::bind(&FsuWeightPool::validate, this, std::placeholders::_1),
    nullptr, memory_ptr);
  elements[id] = {id, memory_ptr, false, 0, 0, mem_data, -1, LoadState::Idle};
  auto &o = exe_order[0];
  order_to_exec_ids[o].insert(id);
  max_exec_id = std::max(max_exec_id, id);
  return mem_data;
}

void FsuWeightPool::clear() {
  deallocate();
  MemoryPool::clear();
}


void FsuWeightPool::validate(unsigned int id) {
  auto validate_start = std::chrono::high_resolution_clock::now();
#if defined(_WIN32)
  SYSTEM_INFO sysInfo;
  GetSystemInfo(&sysInfo);
  auto page_size = sysInfo.dwAllocationGranularity;
#else
  auto page_size = sysconf(_SC_PAGE_SIZE);
#endif

  auto start_offset = elements[id].start_offset;

  size_t off = (start_offset / page_size) * page_size;
  size_t diff = start_offset - off;
  size_t len = elements[id].weight_len + diff;

  char *ptr = static_cast<char *>(
    mmap(nullptr, len, PROT_READ, MAP_PRIVATE, fd, off));

#ifndef _WIN32
  madvise(ptr, len, MADV_SEQUENTIAL);
#endif


  void *now_ptr =
    static_cast<void *>(ptr + diff);
  memcpy(elements[id].memory_ptr, now_ptr, elements[id].weight_len);

  elements[id].mem_data->setAddr((void *)elements[id].memory_ptr);
  elements[id].mem_data->setValid(true);
  elements[id].active = true;
  elements[id].load_state = LoadState::Loaded;
  const auto ret = munmap(ptr, len);
}

bool FsuWeightPool::loadAllinOrder(unsigned int order) {
  auto exec_ids = order_to_exec_ids[order];
  for (auto &id : exec_ids) {
    validate(id);
  }
  return true;
}

void FsuWeightPool::inActive(unsigned int order) {
  auto exec_ids = order_to_exec_ids[order];

  for (auto &id : exec_ids) {
    elements[id].load_task_id = -1;
    elements[id].load_state = LoadState::UnLoaded;
    elements[id].active = false;
  }
}

bool FsuWeightPool::checkAllLoadComplete(unsigned int order) {

  return true;
}

} // namespace nntrainer
