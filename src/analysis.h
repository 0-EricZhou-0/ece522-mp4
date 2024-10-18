#ifndef __ANALYSIS_H__
#define __ANALYSIS_H__

#include "ast.h"
// #include "simulationUtils.h"
#include <string>
#include <vector>
#include <unordered_set>
#include <set>
#include <unordered_map>
#include <assert.h>
#include <map>
#include <queue>


typedef enum {
    Conv2d_Forward, ReLU_Forward, MaxPool2d_Forward, AdaptiveAvgPool2d_Forward, Linear_Forward,
    Dropout_Forward, BatchNorm2d_Forward, Conv2d_Backward_Weight, Conv2d_Backward_Input, Conv2d_Apply_Grad,
    ReLU_Backward, MaxPool2d_Backward, AdaptiveAvgPool2d_Backward, Linear_Backward_Weight, Linear_Backward_Input,
    Linear_Backward_Bias, Linear_Apply_Grad_Bias, Linear_Apply_Grad_Weight, Dropout_Backward, BatchNorm2d_Backward,
    BatchNorm2d_Apply_Grad, LoadData_A0, makeLoss, Add_Forward, Add_MultiGredient, Concat_Forward, Concat_Backward,
    Scale_Forward, Scale_Backward, GatherV2_Forward, GatherV2_Backward, Add_Backward, Divide_Forward, Divide_Backward_A, Divide_Backward_B,
    Multiply_Forward, Multiply_Backward, Power_Forward, Power_Backward, Sqrt_Forward, Sqrt_Backward, SoftmaxBasic_Forward,
    SoftmaxBasic_Backward, Subtract_Forward, Subtract_Backward, Sum_Forward, Sum_Backward, Tanh_Forward, Tanh_Backward,
    BatchMatMul_Forward, BatchMatMul_Backward, Apply_Grad, Erf_Forward, Erf_Backward, NR_kernel_type
} CUDAKernelType;

const std::string print_kerneltype_array [] = {
    "Conv2d_Forward", "ReLU_Forward", "MaxPool2d_Forward", "AdaptiveAvgPool2d_Forward", "Linear_Forward",
    "Dropout_Forward", "BatchNorm2d_Forward", "Conv2d_Backward_Weight", "Conv2d_Backward_Input", "Conv2d_Apply_Grad",
    "ReLU_Backward", "MaxPool2d_Backward", "AdaptiveAvgPool2d_Backward", "Linear_Backward_Weight", "Linear_Backward_Input",
    "Linear_Backward_Bias", "Linear_Apply_Grad_Bias", "Linear_Apply_Grad_Weight", "Dropout_Backward", "BatchNorm2d_Backward",
    "BatchNorm2d_Apply_Grad", "LoadData_A0", "makeLoss", "Add_Forward", "Add_MultiGredient", "Concat_Forward", "Concat_Backward",
    "Scale_Forward", "Scale_Backward", "GatherV2_Forward", "GatherV2_Backward", "Add_Backward", "Divide_Forward", "Divide_Backward_A", "Divide_Backward_B",
    "Multiply_Forward", "Multiply_Backward", "Power_Forward", "Power_Backward", "Sqrt_Forward", "Sqrt_Backward", "SoftmaxBasic_Forward",
    "SoftmaxBasic_Backward", "Subtract_Forward", "Subtract_Backward", "Sum_Forward", "Sum_Backward", "Tanh_Forward", "Tanh_Backward",
    "BatchMatMul_Forward", "BatchMatMul_Backward", "Apply_Grad", "Erf_Forward", "Erf_Backward"
};

extern const std::unordered_map<std::string, CUDAKernelType> kernel_type_revmap;



class Inactive_period;
class Tensor {
    private:
        Tensor();
    public:
        Tensor(long long size, bool glob = false);
        unsigned long getGlobalOffset();
        std::string name() const;
        bool is_alive(int current_kernel) const;
        void print() const;
        void print_liveness();
        void print_inactive_periods();

        int tensor_id;
        long long size_in_byte;  //Aligned with 4KB
        long long raw_size_byte;
        long long address_offset;
        bool is_global_weight;   // If the tensor is a global tensor.
        
        //Following is the assess pattern information, which is not automatically filled with the model graph input.
        int live_interval[2];   //TODO: Important:  Stores liveness information, live_interval[0] = the first kernel that used the tensor; live_interval[1] = (the last kernel that used the tensor) + 1.  Only intermediate tensor has meaningful liveness interval.
        std::vector<Inactive_period*> inactive_periods;  //TODO: Important:  A vector of inactive periods of this tensor. With the start of inactive_period sorted in ascending order

};

class Inactive_period {
  public:
    int kernelLevel_interval[2];      //  kernelLevel_interval[0] =  the first kernel ID that the tensor is inactive,   kernelLevel_interval[1] = (the last kernel ID that the tensor is inactive) + 1;  kernelLevel_interval should at least at least contain one kernel.  e.g., 456----457
    bool is_looped;             // If true, it means that the tensor is a global tensor, and kernelLevel_interval[0] > kernelLevel_interval[1].
    Tensor* the_tensor;

    double time_estimated;   //Important: we provided a compiler pass function to calculate the estimated execution time for every tensors' inactive period length(time)

    Inactive_period(Tensor* t){the_tensor = t; is_looped = false; };
    void print();
};



enum Eviction_P {
    Hot, Medium, Cold, Dead, Invalid
};

const std::string print_eviction_array [4] = {
    "hot", "medium", "cold", "dead"
};


class CUDAKernel {
    public:
        int kernel_id;
        CUDAKernelType type;

        std::unordered_set<Tensor*> inputs;
        std::unordered_set<Tensor*> outputs;
        Tensor* workspace = nullptr;

        /**
         * @brief number of cycles for the kernel to execute assume all the tensors
         * are presented in the GPU memory and ready for computation.
         */
        //Profiled ideal execution time
        long execution_cycles = -1;

        long pf_execution_cycles = -1;
        long input_pf_execution_cycles = -1;


        CUDAKernel(int kernel_id,
                   CUDAKernelType t,
                   std::vector<Tensor*> input_tensor_list,
                   std::vector<Tensor*> output_tensor_list,
                   Tensor* workspace_tensor);

        // Below used by simulators:
        void getRequiredTensors(std::vector<Tensor*> &required_tensors) const;
        void getRequiredTensors(std::unordered_set<Tensor*> &required_tensors) const;
        void getRequiredTensors(std::vector<Tensor*> &required_tensors,
                                std::vector<Tensor*> &required_input_tensors,
                                std::vector<Tensor*> &required_output_tensors) const;
        void getRequiredTensors(std::vector<Tensor*> &required_input_tensors,
                                std::vector<Tensor*> &required_output_tensors,
                                Tensor*& required_workspace_tensor) const;
        void getTensorBreakdown(std::vector<Tensor*> &inputs,
                                std::vector<Tensor*> &weights,
                                std::vector<Tensor*> &intermediates) const;
        void print();
};


class EvictionGuide_Entry {
    public:
        std::unordered_map<Tensor*, Eviction_P> entry;
        std::unordered_map<Tensor*, double> absolute_time_entry;
};




// Important Compiler pass functions:
//TODO:
void tensor_first_pass_liveness_analysis();

//TODO:
void tensor_second_pass_interval_formation();

//Provided
void get_inactive_periods_time();


//TODO:
void scheduling_prefetch();


// void print_GPU_mem_estimation();

void print_GPU_mem_really_in_use();



























#endif