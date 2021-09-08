// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   memory_pool.cpp
 * @date   11 August 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Memory Pool Class
 */

#include <numeric>
#include <vector>

#include <memory_pool.h>
#include <nntrainer_error.h>

namespace nntrainer {

/**
 * @brief Request Memory from memory pool
 * @note start_time is inclusive, but end_time is exclusive
 */
unsigned int MemoryPool::requestMemory(size_t bytes, unsigned int start_time,
                                       unsigned int end_time) {
  if (mem_pool != nullptr)
    throw std::invalid_argument(
      "Deallocate memory pool before requesting more memory");

  if (end_time <= start_time)
    throw std::invalid_argument(
      "Invalid validity range for the requested memory");

  memory_size.push_back(bytes);
  memory_validity.push_back({start_time, end_time});

  /** invalidate min_pool_size if already there */
  min_pool_size = 0;

  return memory_size.size() - 1;
}

/**
 * @brief Planner the layout with memory planner
 *
 * @details The efficiency of the planner is calculated as the ratio of the
 * theoretical minimum memory requirement divided by the memory requirement
 * given by the memory planner.
 *
 * @details planLayout can be called multiple times as this does not perform
 * any allocation but rather just plans the layout and stores the layout.
 * Subsequent call to this function will overwrite any existing layout.
 */
double MemoryPool::planLayout(const MemoryPlanner &planner) {
  if (mem_pool != nullptr)
    /** mem_pool must be deallocated when planLayout is being called */
    throw std::runtime_error("Planning memory layout after allocation");

  if (memory_size.empty())
    throw std::runtime_error("Planning memory layout for empty pool");

  /** calculate min_pool_size if not already calculated */
  if (min_pool_size == 0)
    min_pool_size = calcMinMemoryRequirement();

  pool_size = planner.planLayout(memory_size, memory_validity, memory_offset);
  if (pool_size < min_pool_size || !validateLayout())
    throw std::runtime_error("Planned layout is not feasible");

  return double(pool_size) / double(min_pool_size);
}

/**
 * @brief Do the allocation of memory
 *
 */
void MemoryPool::allocate() {
  if (pool_size == 0)
    throw std::runtime_error("Allocating memory pool with size 0");

  mem_pool = malloc(pool_size);
  if (mem_pool == nullptr)
    throw std::runtime_error("Allocation memory failed");
}

/**
 * @brief Get the allocated memory
 *
 */
void *MemoryPool::getMemory(unsigned int idx) {
  if (mem_pool == nullptr)
    throw std::invalid_argument("Getting memory before allocation");

  return static_cast<char *>(mem_pool) + memory_offset.at(idx);
}

/**
 * @brief Free all the allocated memory
 *
 */
void MemoryPool::deallocate() {
  if (mem_pool != nullptr)
    free(mem_pool);
}

/**
 * @brief Get the maximum real memory requirement
 *
 */
size_t MemoryPool::size() { return pool_size; }

/**
 * @brief Get the minimum theoretical memory requirement
 *
 */
size_t MemoryPool::minMemoryRequirement() {
  if (memory_size.size() && min_pool_size == 0)
    min_pool_size = calcMinMemoryRequirement();

  return min_pool_size;
}

/**
 * @brief Validate the provided layout so that no two memories to be used at
 * overlap interval has overlapping memories
 */
bool MemoryPool::validateLayout() {
  if (memory_offset.size() != memory_size.size())
    return false;

  if (memory_size.empty())
    return pool_size == 0;

  return validateOverflow() && validateOverlap();
}

/**
 * @brief Validate the provided layout does not overflow outside the given
 * size of the memory pool
 */
bool MemoryPool::validateOverflow() {
  for (unsigned int idx = 0; idx < memory_size.size(); idx++)
    if (memory_offset[idx] + memory_size[idx] > pool_size)
      return false;

  return true;
}

/**
 * @brief check if the two given intervals overlap
 *
 * @param s1 start of interval 1
 * @param e1 end of interval 1
 * @param s2 start of interval 2
 * @param e2 end of interval 2
 *
 * @return true if overlap else false
 *
 * @note overlap check assumes are start is inclusive and end is exclusive
 */
template <typename T> static bool overlap(T s1, T e1, T s2, T e2) {
#if DEBUG
  if (e1 <= s1 || e2 <= s2)
    throw std::invalid_argument("Invalid range for intervals in MemoryPool");
#endif

  return !(e1 <= s2 || e2 <= s1)
}

/**
 * @brief Validate the provided layout so that no two memories to be used at
 * overlap interval has overlapping memories
 */
bool MemoryPool::validateOverlap() {
  /** get sorted permutation */
  std::vector<unsigned int> perm = getSortedPermutation();

  /** iterate over sorted array view and check overlap of memories */
  size_t len = perm.size();
  for (unsigned int i = 0; i < len; i++) {
    unsigned int idx = perm[i];
    size_t mem_start = memory_offset[idx], mem_size = memory_size[idx];
    unsigned int valid_start = memory_validity[idx].first,
                 valid_end = memory_validity[idx].second;
    for (unsigned int match = idx + 1; match < len; match++) {
      if (overlap(mem_start, mem_start + mem_size, memory_offset[match],
                  memory_offset[match] + memory_size[match])) {
        /**
         * if the memories given to two requests overlap, then their valid
         * range should not overlap
         */
        if (overlap(valid_start, valid_end, memory_validity[match].first,
                    memory_validity[match].second))
          return false;
      } else {
        /**
         * as the list memories are sorted by offset, we can safely assume that
         * memory allocations after idx will not overlap as well
         */
        break;
      }
    }
  }

  return true;
}

/**
 * @brief Get sorted permuation for the memory requests
 *
 * @details Performs sorting based on the memory overlap using memory offset
 * as the start and the memory offset + memory size as the end of the interval.
 */
std::vector<unsigned int> MemoryPool::getSortedPermutation() {
  std::vector<unsigned int> perm(memory_size.size());
  std::iota(perm.begin(), perm.end(), 0);
  /** sorted by memory_offset first and then memory_offset + memory_size next */
  std::sort(perm.begin(), perm.end(), [&](auto const &idx1, auto const &idx2) {
    if (memory_offset[idx1] == memory_offset[idx2])
      return memory_size[idx1] < memory_size[idx2];

    return memory_offset[idx1] < memory_offset[idx2];
  });

  return perm;
}

/**
 * @brief Calculate the minimum memory requirement for the given memory requests
 *
 * @note This will be theoretical minimum memory requirement ensuring that the
 * memory usages at the same time do not overlap with their validity. This does
 * not consider about the fragmentation which comes from the actual memory
 * layout.
 */
size_t MemoryPool::calcMinMemoryRequirement() {
  auto max_interval =
    *std::max_element(memory_validity.begin(), memory_validity.end(),
                      [](auto const &val1, auto const &val2) {
                        return val1.second < val2.second;
                      });
  unsigned int last_interval = max_interval.second;

  std::vector<size_t> interval_req(last_interval + 1, 0);
  /**
   * @note This method fills requirement for each value in the interval. This is
   * efficient for the current use case as there is going to be atleast 1 new
   * memory request for each interval because each interval is mapped to a node
   * in the graph.
   */
  for (unsigned int idx = 0; idx < memory_size.size(); idx++) {
    for (unsigned int interval = memory_validity[idx].first;
         interval < memory_validity[idx].second; interval++) {
      interval_req[interval] += memory_size[idx];
    }
  }

  return *std::max_element(interval_req.begin(), interval_req.end());
}

} // namespace nntrainer