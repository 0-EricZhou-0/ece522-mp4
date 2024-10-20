
#include "analysis.h"

#include <assert.h>
#include <math.h>
#include <stdlib.h>

#include <algorithm>
#include <fstream>
#include <iostream>

#include "ast.h"
#include "simulationUtils.h"
#include "simulator.h"
#include "printUtils.h"

using Simulator::DataMovementHint;
using Simulator::PageLocation;

extern std::string migration_policy_str;
extern std::string eviction_policy_str;

extern double GPU_frequency_GHz;
extern double GPU_memory_size_GB;
extern double CPU_PCIe_bandwidth_GBps;
extern double SSD_PCIe_bandwidth_GBps;
extern double GPU_malloc_uspB;
// extern double GPU_free_uspB;
extern int prefetch_degree;
extern int borden;
extern int is_transformer;
double CPU_memory_line_GB = -1;
double SSD_latency_us = -1;
double system_latency_us = -1;
double delta_parameter = -1;
double loosen_parameter = 1;

long long memory_offset_intermediate = 0;
long long memory_offset_weights = 0;
int kernel_index = 0;
int prefetch_optimize = 1;
std::vector<Tensor *> tensor_list;
std::vector<CUDAKernel> kernel_list;

// TODO: Important: The global list for all inactive tensor periods
std::vector<InactivePeriod *> inactive_periods_list;

std::vector<double> kernel_time_table;
std::vector<EvictionGuide_Entry> EvictionGuide_Table;
std::vector<long> GPU_resident_memory_estimation;
std::vector<long> CPU_resident_memory_estimation;

std::vector<DataMovementHint> movement_hints;
std::vector<InactivePeriod *> offloaded_local_intervals;

string Tensor::name() const { return "tensor" + std::to_string(tensor_id); }

bool Tensor::is_alive(int current_kernel) const {
  return is_global_weight || (live_interval.second == -1 ? current_kernel == live_interval.first
                                                         : current_kernel >= live_interval.first &&
                                                               current_kernel < live_interval.second);
}

void Tensor::print() const {
  std::cout << "tensor" << tensor_id << " Is weight (global)?: " << this->is_global_weight << ", "
            << "Size in byte: " << size_in_byte << std::endl;
}

Tensor::Tensor(long long size, bool glob) {
  static int tensor_count = 0;
  tensor_id = tensor_count++;
  size_in_byte = size;
  raw_size_byte = size;
  is_global_weight = glob;
  if (glob) {
    address_offset = memory_offset_weights;
    // page-level alignment
    long N_pages = (size % 4096 == 0) ? (size / 4096) : ((size / 4096) + 1);
    memory_offset_weights += N_pages * 4096;
    size_in_byte = N_pages * 4096;
  } else {
    address_offset = memory_offset_intermediate;
    // page-level alignment
    long N_pages = (size % 4096 == 0) ? (size / 4096) : ((size / 4096) + 1);
    memory_offset_intermediate += N_pages * 4096;
    size_in_byte = N_pages * 4096;
  }
}

unsigned long Tensor::getGlobalOffset() {
  return address_offset + (is_global_weight ? 0 : memory_offset_weights);
}

CUDAKernel::CUDAKernel(int kernel_id, CUDAKernelType t, std::vector<Tensor *> input_tensor_list,
                       std::vector<Tensor *> output_tensor_list, Tensor *workspace_tensor) {
  this->kernel_id = kernel_id;
  this->type = t;
  this->inputs.insert(input_tensor_list.begin(), input_tensor_list.end());
  this->outputs.insert(output_tensor_list.begin(), output_tensor_list.end());
  this->workspace = workspace;
}

void CUDAKernel::print() {
  std::cout << "---------------------------------------------------------------"
               "---------------"
            << std::endl;
  std::cout << "Kernel ID: " << kernel_id << ", "
            << "Name: " << print_kerneltype_array[type] << std::endl;
  std::cout << "Execution Time:            " << execution_cycles << std::endl;
  std::cout << "Input Tensors:" << std::endl;
  for (auto it = inputs.begin(); it != inputs.end(); it++) {
    (*it)->print();
  }
  std::cout << "Output Tensors:" << std::endl;
  for (auto it = outputs.begin(); it != outputs.end(); it++) {
    (*it)->print();
  }
}

void CUDAKernel::getRequiredTensors(std::vector<Tensor *> &required_tensors) const {
  std::unordered_set<Tensor *> set;
  getRequiredTensors(set);
  for (Tensor *tensor : set) required_tensors.push_back(tensor);
}

void CUDAKernel::getRequiredTensors(std::unordered_set<Tensor *> &required_tensors) const {
  for (Tensor *tensor : inputs) required_tensors.insert(tensor);
  for (Tensor *tensor : outputs) required_tensors.insert(tensor);
}

void CUDAKernel::getRequiredTensors(std::vector<Tensor *> &required_tensors,
                                    std::vector<Tensor *> &required_input_tensors,
                                    std::vector<Tensor *> &required_output_tensors) const {
  std::unordered_set<Tensor *> set;
  for (Tensor *tensor : inputs) {
    set.insert(tensor);
    required_tensors.push_back(tensor);
    required_input_tensors.push_back(tensor);
  }
  for (Tensor *tensor : outputs) {
    if (set.find(tensor) == set.end()) {
      required_tensors.push_back(tensor);
      required_output_tensors.push_back(tensor);
    }
  }
}

// TODO: You should write a mini compiler pass to fill the liveness information
// of every tensor (liveness information)
void tensor_first_pass_liveness_analysis() {
  const int tensor_num = tensor_list.size();
  const int kernel_num = kernel_list.size();

  for (int i = 0; i < tensor_num; i++) {
    Tensor *current_tensor = tensor_list[i];
    if (!current_tensor->is_global_weight) {  // This tensor is a local one
      // First we need to find its death time:
      bool find = false;
      for (int j = kernel_num - 1; j >= 0; j--) {
        if (kernel_list[j].inputs.find(current_tensor) != kernel_list[j].inputs.end()) {
          find = true;
          current_tensor->live_interval.second = j + 1;
          break;
        }
      }
      if (!find) {
        current_tensor->live_interval.second = -1;
      }
      // Second we need to find its birth time
      for (int j = 0; j < kernel_num; j++) {
        if (kernel_list[j].inputs.find(current_tensor) != kernel_list[j].inputs.end() ||
            kernel_list[j].outputs.find(current_tensor) != kernel_list[j].outputs.end()) {
          current_tensor->live_interval.first = j;
          break;
        }
      }
    }
  }
}

void Tensor::print_liveness() {
  this->print();
  if (!this->is_global_weight) {
    std::cout << "Liveness: Birth: " << this->live_interval.first << ", Death: " << this->live_interval.second
              << "." << std::endl;
  } else {
    std::cout << "Liveness: Global" << std::endl;
  }
}

// TODO: You should write another compiler pass to fill all the inactive periods
// for every tensor (inactive_periods)
void tensor_second_pass_interval_formation() {
  const int tensor_num = tensor_list.size();
  const int kernel_num = kernel_list.size();

  long target_mem_line = (long)(GPU_memory_size_GB * 1024 * 1024 * 1024);

  for (int i = 0; i < tensor_num; i++) {
    Tensor *current_tensor = tensor_list[i];
    if (!current_tensor->is_global_weight) {
      if (current_tensor->live_interval.second != -1) {
        bool a_interval_started = false;
        for (int j = current_tensor->live_interval.first; j < current_tensor->live_interval.second; j++) {
          // j is the current kernel;
          if (kernel_list[j].inputs.find(current_tensor) != kernel_list[j].inputs.end() ||
              kernel_list[j].outputs.find(current_tensor) != kernel_list[j].outputs.end()) {
            if (!a_interval_started) {
              if (j + 1 < current_tensor->live_interval.second &&
                  kernel_list[j + 1].inputs.find(current_tensor) == kernel_list[j + 1].inputs.end() &&
                  kernel_list[j + 1].outputs.find(current_tensor) == kernel_list[j + 1].outputs.end()) {
                // Start one interval
                InactivePeriod *new_interval = new InactivePeriod(current_tensor);
                new_interval->kernelLevel_interval.first = j + 1;
                a_interval_started = true;
                current_tensor->inactive_periods.push_back(new_interval);
                inactive_periods_list.push_back(new_interval);
              }
            } else {
              inactive_periods_list.back()->kernelLevel_interval.second = j;
              a_interval_started = false;

              if (j + 1 < current_tensor->live_interval.second &&
                  kernel_list[j + 1].inputs.find(current_tensor) == kernel_list[j + 1].inputs.end() &&
                  kernel_list[j + 1].outputs.find(current_tensor) == kernel_list[j + 1].outputs.end()) {
                // Start one interval
                InactivePeriod *new_interval = new InactivePeriod(current_tensor);
                new_interval->kernelLevel_interval.first = j + 1;
                a_interval_started = true;
                current_tensor->inactive_periods.push_back(new_interval);
                inactive_periods_list.push_back(new_interval);
              }
            }
          }
        }
        assert(!a_interval_started);
      }
    } else {
      // This tensor is global
      // First find one use
      int one_use = 0;
      for (int j = 0; j < kernel_num; j++) {
        if (kernel_list[j].inputs.find(current_tensor) != kernel_list[j].inputs.end() ||
            kernel_list[j].outputs.find(current_tensor) != kernel_list[j].outputs.end()) {
          one_use = j;
          break;
        }
      }

      bool a_interval_started = false;
      for (int j = one_use; j < kernel_num; j++) {
        // j is the current kernel;
        if (kernel_list[j].inputs.find(current_tensor) != kernel_list[j].inputs.end() ||
            kernel_list[j].outputs.find(current_tensor) != kernel_list[j].outputs.end()) {
          if (!a_interval_started) {
            if (j + 1 < kernel_num &&
                kernel_list[j + 1].inputs.find(current_tensor) == kernel_list[j + 1].inputs.end() &&
                kernel_list[j + 1].outputs.find(current_tensor) == kernel_list[j + 1].outputs.end()) {
              // Start one interval
              InactivePeriod *new_interval = new InactivePeriod(current_tensor);
              new_interval->kernelLevel_interval.first = j + 1;
              a_interval_started = true;
              current_tensor->inactive_periods.push_back(new_interval);
              inactive_periods_list.push_back(new_interval);
            } else if (j + 1 == kernel_num &&
                       kernel_list[0].inputs.find(current_tensor) == kernel_list[0].inputs.end() &&
                       kernel_list[0].outputs.find(current_tensor) == kernel_list[0].outputs.end()) {
              // Start one interval
              InactivePeriod *new_interval = new InactivePeriod(current_tensor);
              new_interval->kernelLevel_interval.first = 0;
              a_interval_started = true;
              current_tensor->inactive_periods.push_back(new_interval);
              inactive_periods_list.push_back(new_interval);
            }
          } else {
            inactive_periods_list.back()->kernelLevel_interval.second = j;
            a_interval_started = false;

            if (j + 1 < kernel_num &&
                kernel_list[j + 1].inputs.find(current_tensor) == kernel_list[j + 1].inputs.end() &&
                kernel_list[j + 1].outputs.find(current_tensor) == kernel_list[j + 1].outputs.end()) {
              // Start one interval
              InactivePeriod *new_interval = new InactivePeriod(current_tensor);
              new_interval->kernelLevel_interval.first = j + 1;
              a_interval_started = true;
              current_tensor->inactive_periods.push_back(new_interval);
              inactive_periods_list.push_back(new_interval);
            } else if (j + 1 == kernel_num &&
                       kernel_list[0].inputs.find(current_tensor) == kernel_list[0].inputs.end() &&
                       kernel_list[0].outputs.find(current_tensor) == kernel_list[0].outputs.end()) {
              // Start one interval
              InactivePeriod *new_interval = new InactivePeriod(current_tensor);
              new_interval->kernelLevel_interval.first = 0;
              a_interval_started = true;
              current_tensor->inactive_periods.push_back(new_interval);
              inactive_periods_list.push_back(new_interval);
            }
          }
        }
      }
      for (int j = 0; j <= one_use; j++) {
        // j is the current kernel;
        if (kernel_list[j].inputs.find(current_tensor) != kernel_list[j].inputs.end() ||
            kernel_list[j].outputs.find(current_tensor) != kernel_list[j].outputs.end()) {
          if (!a_interval_started) {
            if (j < one_use &&
                kernel_list[j + 1].inputs.find(current_tensor) == kernel_list[j + 1].inputs.end() &&
                kernel_list[j + 1].outputs.find(current_tensor) == kernel_list[j + 1].outputs.end()) {
              // Start one interval
              InactivePeriod *new_interval = new InactivePeriod(current_tensor);
              new_interval->kernelLevel_interval.first = j + 1;
              a_interval_started = true;
              current_tensor->inactive_periods.push_back(new_interval);
              inactive_periods_list.push_back(new_interval);
            }
          } else {
            inactive_periods_list.back()->kernelLevel_interval.second = j;
            if (j < inactive_periods_list.back()->kernelLevel_interval.first) {
              inactive_periods_list.back()->is_looped = 1;
            }
            a_interval_started = false;

            if (j < one_use &&
                kernel_list[j + 1].inputs.find(current_tensor) == kernel_list[j + 1].inputs.end() &&
                kernel_list[j + 1].outputs.find(current_tensor) == kernel_list[j + 1].outputs.end()) {
              // Start one interval
              InactivePeriod *new_interval = new InactivePeriod(current_tensor);
              new_interval->kernelLevel_interval.first = j + 1;
              a_interval_started = true;
              current_tensor->inactive_periods.push_back(new_interval);
              inactive_periods_list.push_back(new_interval);
            }
          }
        }
      }

      assert(!a_interval_started);
    }
  }
}

void Tensor::print_inactive_periods() {
  // print();
  std::cout << "Inactive Periods:" << std::endl;
  for (int i = 0; i < inactive_periods.size(); i++) {
    std::cout << "interval " << i << ": " << inactive_periods[i]->kernelLevel_interval.first << "--------"
              << inactive_periods[i]->kernelLevel_interval.second << std::endl;
    std::cout << "Estimated Time:" << inactive_periods[i]->time_estimated << std::endl;
  }
  std::cout << "_______________________________________________________________" << std::endl;
}

// A provided compiler pass to calculate the estimated execution time for every
// tensors' inactive period length(time)
void get_inactive_periods_time() {
  int kernel_num = kernel_list.size();

  // Setup a cumulative time list;
  double time = 0;
  kernel_time_table.push_back(0);
  for (int i = 0; i < kernel_num; i++) {
    time += (double)kernel_list[i].execution_cycles / (double)(GPU_frequency_GHz * 1000);
    kernel_time_table.push_back(time);
  }

  // Fill the looped extend kernel time table      0 - 2 * kernel_num
  std::vector<double> kernel_time_table_extended;
  kernel_time_table_extended.resize(kernel_num);
  for (int j = 0; j < kernel_num; j++) {
    kernel_time_table_extended[j] = kernel_time_table[j];
  }
  double last_time = kernel_time_table[kernel_num];
  kernel_time_table_extended.push_back(last_time);
  for (int j = 0; j < kernel_num; j++) {
    last_time += (double)kernel_list[j].execution_cycles / (double)(GPU_frequency_GHz * 1000);
    kernel_time_table_extended.push_back(last_time);
  }

  for (int i = 0; i < inactive_periods_list.size(); i++) {
    if (!inactive_periods_list[i]->is_looped) {
      assert(inactive_periods_list[i]->kernelLevel_interval.second >
             inactive_periods_list[i]->kernelLevel_interval.first);
      inactive_periods_list[i]->time_estimated =
          kernel_time_table[inactive_periods_list[i]->kernelLevel_interval.second] -
          kernel_time_table[inactive_periods_list[i]->kernelLevel_interval.first];
    } else {
      assert(inactive_periods_list[i]->kernelLevel_interval.second <
             inactive_periods_list[i]->kernelLevel_interval.first);
      int end = inactive_periods_list[i]->kernelLevel_interval.second;
      int start = inactive_periods_list[i]->kernelLevel_interval.first;
      end += kernel_num;
      inactive_periods_list[i]->time_estimated =
          kernel_time_table_extended[end] - kernel_time_table_extended[start];
    }
  }
}

void InactivePeriod::print() {
  std::cout << "interval " << ": " << kernelLevel_interval.first << "--------" << kernelLevel_interval.second
            << std::endl;
  std::cout << "Estimated Time:" << time_estimated << std::endl;
  std::cout << "Tensor: ";
  this->tensor_back_ptr->print();
  std::cout << "_______________________________________________________________" << std::endl;
}

void print_GPU_mem_really_in_use() {
  for (int i = 0; i < kernel_list.size(); i++) {
    std::vector<Tensor *> r;
    kernel_list[i].getRequiredTensors(r);
    long size_bite = 0;
    for (int j = 0; j < r.size(); j++) {
      size_bite += r[j]->size_in_byte;
    }
    std::cout << "Kernel " << i << ": " << size_bite << std::endl;
  }
}

/**
 * @brief fill this function to schedule your movement hints
 */
void scheduling_movement_hints() {
  // TODO: fill the data structure "std::vector<DataMovementHint> movement_hints" with your own hints!
  iprintf("TODO: schedule your movement hints here\n", "");
}
