/* File: main.cc
 * -------------
 * This file defines the main() routine for the program and not much else.
 * You should not need to modify this file.
 */

#include <chrono>
#include <string>
#include <math.h>
#include <random>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <ctype.h>
#include <iostream>
#include <unistd.h>
#include "utility.h"
#include "errors.h"
#include "parser.h"
#include "y.tab.h"
#include "analysis.h"
#include "codegen.h"
#include "simulationComponents.h"
#include "simulator.h"
#include "printUtils.h"

#define YYDEBUG 1

using std::chrono::duration;
using std::chrono::high_resolution_clock;

// Codegen param
extern int is_resnet;
extern int is_inception;
extern int is_senet;
extern int batch_size;
extern int input_H;
extern int input_W;
extern int num_threads;
extern bool is_individual;
extern bool is_input_pf_only;

// CPU sim param
extern double CPU_PCIe_bandwidth_GBps;
// GPU sim param
extern double GPU_PCIe_bandwidth_GBps;
extern double GPU_frequency_GHz;
extern double GPU_memory_size_GB;
extern double GPU_malloc_uspB;
extern double GPU_free_uspB; // NOT USED FOR NOW
// SSD sim param
extern double SSD_PCIe_bandwidth_GBps;
// PCIe sim param
extern double PCIe_latency_us;  // NOT USED FOR NOW
extern int PCIe_batch_size_in_page;
// Other sim param
extern bool use_prefetch;
extern std::string migration_policy_str;
extern std::string eviction_policy_str;
extern Simulator::MigPolicy migration_policy;
extern Simulator::GPUPageTable::EvcPolicy eviction_policy;
extern int prefetch_degree;
extern int num_candidate;
extern double system_latency_us; // NOT USED FOR NOW

// Other param
//   In codegen, is_UVM specifies whether to use cudaMallocManaged
//   In simulation, is_UVM specifies whether setup is ideal (i.e. all tensor in GPU mem)
bool is_UVM = true;
//   In codegen, num_iteration specifies number of iterations to profile
//   In simulation, num_iteration specifies number of iterations to run
int num_iteration = -1;
int is_transformer = -1;
int borden = 184;

// 
extern double CPU_memory_line_GB;
extern double SSD_read_latency_us;
extern double SSD_write_latency_us;
extern double SSD_latency_us; // Upper bound
extern double delta_parameter;

// Tensor configurations
extern long long memory_offset_intermediate;
extern long long memory_offset_weights;

// 
extern std::vector<Model_Layer*> forward_layers;
extern std::vector<Model_OP*> forward_ops;
extern std::vector<CUDAKernel> kernel_list;
extern std::vector<Tensor*> tensor_list;
extern std::vector<Hidding_Interval*> interval_list;
extern std::vector<EvictionGuide_Entry> EvictionGuide_Table;
extern std::vector<long> GPU_resident_memory_estimation;
extern std::vector<double> kernel_time_table;

// output specifications
std::string nn_model_input_file;
std::string orig_kernel_time_file;
std::string input_pf_kernel_time_file;
std::string workspace_size_file;
std::string pf_kernel_time_file;

std::string tensor_info_file;
std::string kernel_info_file;
std::string kernel_aux_time_file;

std::string stat_output_file;
std::string output_folder_name;
// simulation switches
bool is_simulation = true;
bool output_override = false;
// profiling switches
bool is_compile = true;
bool is_run = true;
int compile_max_thread_num = -1;
bool is_cudnn = false;

// random devices
std::mt19937 rand_device;
double kernel_time_std_dev = 0;
unsigned int ran_seed = 1;
double kernel_speedup = 1;

class RedirStdOut {
    public:
        RedirStdOut(std::string filename) {
            info_file = output_folder_name + "/statistics/" + filename;
            buffer.str("");
            old_cout_buf = std::cout.rdbuf();
            cout_buf = std::cout.rdbuf(buffer.rdbuf());
            printf("Saving %s\n", filename.c_str());
        }
        ~RedirStdOut() {
            std::ofstream fout(info_file.c_str());
            fout << buffer.str();
            fout.close();
            std::cout.rdbuf(old_cout_buf);
        }
    private:
        std::string info_file;
        std::stringstream buffer;
        std::streambuf *old_cout_buf;
        std::streambuf *cout_buf;
};

/* Function: PrintOneToken()
 * Usage: PrintOneToken(T_Double, "3.5", val, loc);
 * -----------------------------------------------
 */
static void PrintOneToken(yytokentype token, const char *text, YYSTYPE value,
                          yyltype loc)
{
  char buffer[] = {'\'', (char) token, '\'', '\0'};
  const char *name = token >= T_Sequential ? gTokenNames[token - T_Sequential] : buffer;

  printf("%-12s line %d cols %d-%d is %s ", text,
	   loc.first_line, loc.first_column, loc.last_column, name);

  switch(token) {
    case T_IntConstant:
      printf("(value = %d)\n", value.integerConstant); break;
    case T_DoubleConstant:
      printf("(value = %g)\n", value.doubleConstant); break;
    case T_BoolConstant:
      printf("(value = %s)\n", value.boolConstant ? "true" : "false"); break;
    case T_Identifier:
	if (strcmp(text, value.identifier)) {
	  printf("(truncated to %s)\n", value.identifier);
	  break;
	}
    default:
      printf("\n"); break;
  }
}

void CheckVar(double var, std::string variable_name, bool gt=true) {
    if ((gt && var < 0) || (!gt && var > 0)) {
        eprintf("Invalid or missing <%s>, current value: %f, should be %s than 0, aborting\n",
                variable_name.c_str(), var, gt ? "greater" : "less");
        Assert(false);
    }
}

void SimulationParamSanityCheck() {
    // parameter validation (existence)
    CheckVar(PCIe_batch_size_in_page, "PCIe_batch_size_in_page");
    CheckVar(CPU_PCIe_bandwidth_GBps, "CPU_PCIe_bandwidth_GBps");
    CheckVar(GPU_PCIe_bandwidth_GBps, "GPU_PCIe_bandwidth_GBps");
    CheckVar(SSD_PCIe_bandwidth_GBps, "SSD_PCIe_bandwidth_GBps");
    CheckVar(GPU_frequency_GHz, "GPU_frequency_GHz");
    CheckVar(GPU_memory_size_GB, "GPU_memory_size_GB");
    CheckVar(GPU_malloc_uspB, "GPU_malloc_uspB");
    CheckVar(GPU_free_uspB, "GPU_free_uspB");
    CheckVar(CPU_memory_line_GB, "CPU_memory_line_GB");

    if (migration_policy == Simulator::MigPolicy::DEEPUM)
        Assert(eviction_policy == Simulator::GPUPageTable::EvcPolicy::DEEPUM);
    if (eviction_policy == Simulator::GPUPageTable::EvcPolicy::DEEPUM)
        Assert(migration_policy == Simulator::MigPolicy::DEEPUM);
    if (migration_policy == Simulator::MigPolicy::DEEPUM)
        CheckVar(prefetch_degree, "prefetch_degree");
    else
        CheckVar(prefetch_degree, "prefetch_degree", false);
    if (eviction_policy == Simulator::GPUPageTable::EvcPolicy::GUIDED)
        CheckVar(num_candidate, "num_candidate");
    else
        CheckVar(num_candidate, "num_candidate", false);
    CheckVar(num_iteration, "num_iteration");

    // parameter validation (value)
    if (SSD_PCIe_bandwidth_GBps > GPU_PCIe_bandwidth_GBps) {
        eprintf("Invalid SSD Bandwidth [%f] > GPU Bandwidth [%f]\n",
                SSD_PCIe_bandwidth_GBps, GPU_PCIe_bandwidth_GBps);
        Assert(false);
    }
    if (CPU_PCIe_bandwidth_GBps > GPU_PCIe_bandwidth_GBps) {
        eprintf("Invalid CPU Bandwidth [%f] > GPU Bandwidth [%f]\n",
                SSD_PCIe_bandwidth_GBps, GPU_PCIe_bandwidth_GBps);
        Assert(false);
    }
    if (SSD_PCIe_bandwidth_GBps > CPU_PCIe_bandwidth_GBps) {
        eprintf("Unsupported SSD Bandwidth [%f] > CPU Bandwidth [%f]\n",
                SSD_PCIe_bandwidth_GBps, CPU_PCIe_bandwidth_GBps);
        Assert(false);
    }
    if (GPU_PCIe_bandwidth_GBps > SSD_PCIe_bandwidth_GBps + CPU_PCIe_bandwidth_GBps) {
        eprintf("Unsupported GPU Bandwidth [%f] > SSD Bandwidth [%f] + CPU Bandwidth [%f]\n",
                GPU_PCIe_bandwidth_GBps, SSD_PCIe_bandwidth_GBps, CPU_PCIe_bandwidth_GBps);
        Assert(false);
    }
    if (kernel_speedup <= 0) {
        eprintf("Invalid kernel speedup [%f]\n", kernel_speedup);
        Assert(false);
    }
}

void SetupOutputFolder() {
    if (output_override)
        wprintf("Overriding output folder <%s>...\n", output_folder_name.c_str());
    Assert(system(("mkdir -p " + output_folder_name).c_str()) == 0);
    Assert(system(("find " + output_folder_name + "/statistics -name \"*.config\" -type f | xargs rm -f").c_str()) == 0);
    // clean up dirs
    if (output_override && !is_simulation) {
        Assert(system(("rm -rf " + output_folder_name + "/include").c_str()) == 0);
        Assert(system(("rm -rf " + output_folder_name + "/src").c_str()) == 0);
        Assert(system(("rm -rf " + output_folder_name + "/bin").c_str()) == 0);
        Assert(system(("rm -rf " + output_folder_name + "/scripts").c_str()) == 0);
        Assert(system(("rm -rf " + output_folder_name + "/profiling_src").c_str()) == 0);
        Assert(system(("rm -f " + output_folder_name + "/main.cu").c_str()) == 0);
        Assert(system(("rm -f " + output_folder_name + "/main").c_str()) == 0);
    }
    // make dirs
    Assert(system(("mkdir -p " + output_folder_name + "/statistics").c_str()) == 0);
    // LRU visualization ////////////////////////////////////////////////////////////
    // Assert(system(("mkdir -p " + output_folder_name + "/lru_trace").c_str()) == 0);
    /////////////////////////////////////////////////////////////////////////////////
    if (!is_simulation) {
        Assert(system(("mkdir -p " + output_folder_name + "/include").c_str()) == 0);
        Assert(system(("mkdir -p " + output_folder_name + "/src").c_str()) == 0);
        Assert(system(("mkdir -p " + output_folder_name + "/bin").c_str()) == 0);
        Assert(system(("mkdir -p " + output_folder_name + "/scripts").c_str()) == 0);
        Assert(system(("mkdir -p " + output_folder_name + "/profiling_src").c_str()) == 0);
        Assert(system(("cp ./resources/cudadnnUtil.cuh " + output_folder_name + "/include/cudadnnUtil.cuh").c_str()) == 0);
        Assert(system(("cp ./resources/cudadnnUtil.cu " + output_folder_name + "/src/cudadnnUtil.cu").c_str()) == 0);
        Assert(system(("cp ./resources/Makefile " + output_folder_name + "/Makefile").c_str()) == 0);
        if (is_individual) {
            Assert(system(("cp ./resources/compileAndRunI.sh " + output_folder_name + "/scripts/compileAndRun.sh").c_str()) == 0);
        } else {
            Assert(system(("cp ./resources/compileAndRunW.sh " + output_folder_name + "/scripts/compileAndRun.sh").c_str()) == 0);
        }
    }
}

void exportKernelInfo(string dnn_type, int batch_size) {
    string output_path = "../results_modified/" + dnn_type + "/sim_input/";
    Assert(system(("mkdir -p " + output_path).c_str()) == 0);
    double GPU_frequency_Hz = GPU_frequency_GHz * pow(10, 9);
    FILE* fout;

    fout = fopen((output_path + to_string(batch_size) + "Tensor.info").c_str(), "w");
    Assert(fout);
    for (Tensor* tensor : tensor_list) {
        fprintf(fout, "%d %lld %s\n", tensor->tensor_id, tensor->raw_size_byte, tensor->is_global_weight ? "true" : "false");
    }
    fflush(fout);
    fclose(fout);

    fout = fopen((output_path + to_string(batch_size) + "Kernel.info").c_str(), "w");
    Assert(fout);
    for (CUDAKernel &kernel : kernel_list) {
        std::vector<Tensor*> required_input_tensors;
        std::vector<Tensor*> required_output_tensors;
        Tensor* required_workspace_tensor;

        kernel.getRequiredTensors(required_input_tensors, required_output_tensors, required_workspace_tensor);

        int item_cnt = 0;
        fprintf(fout, "%d %s %.6f [", kernel.kernel_id, print_kerneltype_array[kernel.type].c_str(),
            kernel.execution_cycles / GPU_frequency_Hz * 1000);
        item_cnt = 0;
        for (Tensor *t : required_input_tensors) {
            fprintf(fout, "%s%d", item_cnt ? "," : "", t->tensor_id);
            item_cnt++;
        }
        fprintf(fout, "] [");
        item_cnt = 0;
        for (Tensor *t : required_output_tensors) {
            fprintf(fout, "%s%d", item_cnt ? "," : "", t->tensor_id);
            item_cnt++;
        }
        if (required_workspace_tensor) {
            fprintf(fout, "] %d\n", required_workspace_tensor->tensor_id);
        } else {
            fprintf(fout, "]\n");
        }
    }
    fflush(fout);
    fclose(fout);

    fout = fopen((output_path + to_string(batch_size) + "AuxTime.info").c_str(), "w");
    Assert(fout);
    for (CUDAKernel &kernel : kernel_list) {
        fprintf(fout, "%d %.6f %.6f\n", kernel.kernel_id,
            (double) kernel.input_pf_execution_cycles / GPU_frequency_Hz * 1000,
            (double) kernel.pf_execution_cycles / GPU_frequency_Hz * 1000);
    }
    fflush(fout);
    fclose(fout);
}

void loadKernelInfo() {
    double GPU_frequency_Hz = GPU_frequency_GHz * pow(10, 9);
    std::string line;

    {
        // load tensor info
        std::ifstream tensor_info_fin(tensor_info_file);
        Assert(tensor_info_fin.good());
        iprintf("Loading tensor info from file <%s>\n", tensor_info_file.c_str());

        string tensor_id, tensor_size, tensor_global;
        tensor_list.clear();
        while (std::getline(tensor_info_fin, line)) {
            std::stringstream ss(line);
            tensor_id.clear();
            ss >> tensor_id >> tensor_size >> tensor_global;
            if (!tensor_id.size()) continue;

            // populate tensor list
            tensor_list.push_back(new Tensor(
                std::stoll(tensor_size), tensor_global == "true"
            ));
        }
        iprintf("Total %d tensors found\n", tensor_list.size());
    } {
        // load kernel info
        std::ifstream kinfo_fin(kernel_info_file);
        Assert(kinfo_fin.good());
        iprintf("Loading kernel info from file <%s>\n", kernel_info_file.c_str());

        string kernel_idx, ktype, exe_time, input_tensor_list, output_tensor_list, workspace;
        kernel_list.clear();
        while (std::getline(kinfo_fin, line)) {
            std::stringstream ss(line);
            kernel_idx.clear();
            ss >> kernel_idx >> ktype >> exe_time >> input_tensor_list >> output_tensor_list >> workspace;
            if (!kernel_idx.size()) continue;

            std::vector<Tensor*> inputs;
            std::vector<Tensor*> outputs;
            Tensor* workspace_tensor;

            auto list_to_vec_tensor = [](std::vector<Tensor*> &vec_tensor, const std::string &str_list) {
                Assert(str_list.front() == '[' && str_list.back() == ']');
                std::string tensor_id;
                std::istringstream ss(str_list.substr(1, str_list.size() - 2));
                while(std::getline(ss, tensor_id, ',')) {
                    int tensor_id_i = std::stoi(tensor_id);
                    Assert(tensor_id_i < tensor_list.size());
                    vec_tensor.push_back(tensor_list[tensor_id_i]);
                }
            };
            // input tensor list
            list_to_vec_tensor(inputs, input_tensor_list);
            // output tensor list
            list_to_vec_tensor(outputs, output_tensor_list);
            // workspace tensor
            Assert(std::stoi(workspace) < tensor_list.size());
            workspace_tensor = workspace.size() ? tensor_list[std::stoi(workspace)] : nullptr;
            if (workspace_tensor) outputs.push_back(workspace_tensor);

            // populate kernel list
            kernel_list.push_back(CUDAKernel(
                std::stoi(kernel_idx), kernel_type_revmap.at(ktype), inputs, outputs, workspace_tensor
            ));
            kernel_list.back().execution_cycles = std::stod(exe_time) * GPU_frequency_Hz / 1000.0;
        }
        iprintf("Total %d kernels found\n", kernel_list.size());
    } {
        // load kernel aux info
        std::ifstream auxtime_fin(kernel_aux_time_file);
        Assert(auxtime_fin.good());
        iprintf("Loading aux kernel info from file <%s>\n", kernel_aux_time_file.c_str());

        string kernel_idx, input_pf_exe_time, pf_exe_time;
        while (std::getline(auxtime_fin, line)) {
            std::stringstream ss(line);
            kernel_idx.clear();
            ss >> kernel_idx >> input_pf_exe_time >> pf_exe_time;
            if (!kernel_idx.size()) continue;

            // populate aux timing info for performance model
            int kernel_idx_i = std::stoi(kernel_idx);
            kernel_list[kernel_idx_i].input_pf_execution_cycles = std::stod(input_pf_exe_time) * GPU_frequency_Hz / 1000.0;
            kernel_list[kernel_idx_i].pf_execution_cycles = std::stod(pf_exe_time) * GPU_frequency_Hz / 1000.0;
        }
    } {
        // validation
        for (unsigned tensor_idx = 0; tensor_idx < kernel_list.size(); tensor_idx++) {
            Assert(tensor_list[tensor_idx]->tensor_id == tensor_idx);
        }
        for (unsigned kernel_idx = 0; kernel_idx < kernel_list.size(); kernel_idx++) {
            Assert(kernel_list[kernel_idx].kernel_id == kernel_idx);
            Assert(kernel_list[kernel_idx].input_pf_execution_cycles >= kernel_list[kernel_idx].execution_cycles);
            Assert(kernel_list[kernel_idx].pf_execution_cycles >= kernel_list[kernel_idx].input_pf_execution_cycles);
        }
    }
}

void loadWorkspaceSizes() {
    std::ifstream wok_f(workspace_size_file);
    Assert(wok_f.good());

    int kernel_id;
    string workspace_size_str;
    string unit;
    size_t workspace_size;
    iprintf("Loading workspace sizes from file <%s> for %d kernels\n",
            workspace_size_file.c_str(), kernel_list.size());
    for (int i = 0; i < kernel_list.size(); i++) {
        workspace_size_str = "";
        wok_f >> kernel_id >> workspace_size_str >> unit;
        Assert(kernel_id == i);
        Assert(workspace_size_str != "");
        Assert(unit == "B");
        workspace_size = std::stoull(workspace_size_str);
        Assert(workspace_size >= 0);
        if (workspace_size > 0) {
            kernel_list[i].workspace = new Tensor(workspace_size, false);
            kernel_list[i].outputs.insert(kernel_list[i].workspace);
            tensor_list.push_back(kernel_list[i].workspace);
        }
    }
    iprintf("Loading workspace sizes done\n", "");
}

void loadKernelTimes() {
    double GPU_frequency_Hz = GPU_frequency_GHz * pow(10, 9);

    std::ifstream orig_f(orig_kernel_time_file);
    std::ifstream pf_f(pf_kernel_time_file);
    std::ifstream inputpf_f(input_pf_kernel_time_file);
    Assert(orig_f.good());
    Assert(pf_f.good());
    Assert(inputpf_f.good());

    int kernel_num;
    string exe_time_ms_str;
    string unit;
    // read in all the execution times
    long exe_time_cycle;
    double total_time = 0, pf_total_time = 0, input_pf_total_time = 0;
    unsigned long total_time_cycle = 0, pf_total_time_cycle = 0, input_pf_total_time_cycle = 0;
    iprintf("Loading kernel times from file <%s> and <%s> and <%s> for %d kernels\n",
            orig_kernel_time_file.c_str(), pf_kernel_time_file.c_str(), input_pf_kernel_time_file.c_str(), kernel_list.size());
    if (kernel_speedup != 1) {
        iprintf("Using kernel speedup of %.4fx\n", kernel_speedup);
    }
    for (int i = 0; i < kernel_list.size(); i++) {
        double delta_execution_time;
        // read in ideal execution time from file
        exe_time_ms_str.clear();
        orig_f >> kernel_num >> exe_time_ms_str >> unit;
        Assert(kernel_num == i);
        Assert(exe_time_ms_str != "");
        Assert(unit == "ms");
        exe_time_cycle = std::stod(exe_time_ms_str) * GPU_frequency_Hz / 1000.0;
        delta_execution_time = exe_time_cycle - exe_time_cycle / kernel_speedup;
        kernel_list[i].execution_cycles = exe_time_cycle - delta_execution_time;
        Assert(kernel_list[i].execution_cycles > 0);
        total_time += kernel_list[i].execution_cycles / GPU_frequency_Hz * 1000;
        total_time_cycle += exe_time_cycle;
        // read in input_pf execution time from file
        exe_time_ms_str.clear();
        inputpf_f >> kernel_num >> exe_time_ms_str >> unit;
        Assert(kernel_num == i);
        Assert(exe_time_ms_str != "");
        Assert(unit == "ms");
        exe_time_cycle = std::stod(exe_time_ms_str) * GPU_frequency_Hz / 1000.0;
        kernel_list[i].input_pf_execution_cycles = exe_time_cycle - delta_execution_time;
        if (kernel_list[i].input_pf_execution_cycles < kernel_list[i].execution_cycles)
            kernel_list[i].input_pf_execution_cycles = kernel_list[i].execution_cycles;
        // read in pf execution time from file
        exe_time_ms_str.clear();
        pf_f >> kernel_num >> exe_time_ms_str >> unit;
        Assert(kernel_num == i);
        Assert(exe_time_ms_str != "");
        Assert(unit == "ms");
        exe_time_cycle = std::stod(exe_time_ms_str) * GPU_frequency_Hz / 1000.0;
        kernel_list[i].pf_execution_cycles = exe_time_cycle - delta_execution_time;
        if (kernel_list[i].pf_execution_cycles < kernel_list[i].input_pf_execution_cycles)
            kernel_list[i].pf_execution_cycles = kernel_list[i].input_pf_execution_cycles;
        Assert(kernel_list[i].pf_execution_cycles > 0);
        Assert(exe_time_cycle > 0);
        pf_total_time += kernel_list[i].pf_execution_cycles / GPU_frequency_Hz * 1000;
        pf_total_time_cycle += exe_time_cycle;
        Assert(kernel_list[i].input_pf_execution_cycles > 0);
    }
    nprintf("Total time (Ideal): %f ms %lu cycles; (PF): %f ms %lu cycles\n", 
            total_time, total_time_cycle, pf_total_time, pf_total_time_cycle);
    // make sure kernel times file have no other entries left
    exe_time_ms_str = "";
    orig_f >> exe_time_ms_str;
    Assert(exe_time_ms_str == "");
    // make sure pf kernel times file have no other entries left
    exe_time_ms_str = "";
    pf_f >> exe_time_ms_str;
    Assert(exe_time_ms_str == "");
    // make sure inputpf kernel times file have no other entries left
    exe_time_ms_str = "";
    inputpf_f >> exe_time_ms_str;
    Assert(exe_time_ms_str == "");
    iprintf("Loading kernel times done\n", "");
}

/* Function: main()
 * ----------------
 * Entry point to the entire program.  We parse the command line and turn
 * on any debugging flags requested by the user when invoking the program.
 * InitScanner() is used to set up the scanner.
 * InitParser() is used to set up the parser. The call to yyparse() will
 * attempt to parse a complete program from the input.
 */
int main(int argc, char *argv[]) {
    // config file should be the first argument
    if (argc == 1) {
        eprintf("Please specify a config file\n", "");
        Assert(false);
    }
    // exit if config file does not exist
    std::string config_file_path = string(argv[1]);
    std::ifstream config_file(config_file_path);
    if (!config_file.good()) {
        eprintf("Config file <%s> does not exist\n", argv[1]);
        Assert(false);
    }
    // parse config file
    std::string line;
    std::string command;
    std::string value;
    printf("\nConfigs:\n");
    while (std::getline(config_file, line)) {
        std::stringstream ss(line);
        command.clear();
        value.clear();
        ss >> command >> value;
        if (command != "#" && command != "")
            printf("%27s: <%s>\n", command.c_str(), value.c_str());

        // general settings
        if (command == "output_folder")                 { output_folder_name = value; }
        else if (command == "output_override")          { output_override = std::stoi(value) != 0; }
        else if (command == "is_simulation")            { is_simulation = std::stoi(value) != 0; }
        else if (command == "is_profiling")             { is_simulation = std::stoi(value) == 0; }
        // profiling general settings
        else if (command == "is_individual")            { is_individual = std::stoi(value) != 0; }
        else if (command == "is_compile")               { is_compile = std::stoi(value) != 0; }
        else if (command == "compile_max_thread_num")   { compile_max_thread_num = std::stoi(value); }
        else if (command == "is_run")                   { is_run = std::stoi(value) != 0; }
        else if (command == "is_cudnn")                 { is_cudnn = std::stoi(value) != 0; }
        // codegen settings
        else if (command == "is_resnet")                { is_resnet = std::stoul(value); }
        else if (command == "is_inception")             { is_inception = std::stoul(value);}
        else if (command == "is_senet")                 { is_senet = std::stoul(value);}
        else if (command == "is_transformer")           { is_transformer = std::stoi(value); }
        else if (command == "trans_borden")             { borden = std::stoi(value); }
        else if (command == "batch_size")               { batch_size = std::stoi(value); }
        else if (command == "input_H")                  { input_H = std::stoi(value); }
        else if (command == "input_W")                  { input_W = std::stoi(value); }
        else if (command == "num_iteration")            { num_iteration = std::stoi(value); }
        else if (command == "num_threads")              { num_threads = std::stoi(value); }
        else if (command == "is_input_pf_only")         { is_input_pf_only = std::stoi(value) != 0; }
        // simulation general settings
        else if (command == "is_UVM")                   { is_UVM = std::stoi(value) != 0; }
        else if (command == "use_prefetch")             { use_prefetch = std::stoi(value) != 0; }

        else if (command == "nn_model_input_file")      { nn_model_input_file = value; }
        else if (command == "orig_kernel_time_file")    { orig_kernel_time_file = value; }
        else if (command == "workspace_size_file")      { workspace_size_file = value; }
        else if (command == "input_pf_kernel_time_file"){ input_pf_kernel_time_file = value; }
        else if (command == "pf_kernel_time_file")      { pf_kernel_time_file = value; }

        else if (command == "tensor_info_file")         { tensor_info_file = value; }
        else if (command == "kernel_info_file")         { kernel_info_file = value; }
        else if (command == "kernel_aux_time_file")     { kernel_aux_time_file = value; }

        else if (command == "stat_output_file")         { stat_output_file = value; }
        else if (command == "migration_policy")         { migration_policy_str = value; }
        else if (command == "eviction_policy")          { eviction_policy_str = value; }
        else if (command == "num_candidate")            { num_candidate = std::stoul(value); }
        else if (command == "prefetch_degree")          { prefetch_degree = std::stoi(value); }
        else if (command == "delta_parameter")          { delta_parameter = std::stod(value); }
        else if (command == "system_latency_us")        { system_latency_us = std::stod(value); }
        // simulation CPU statistics
        else if (command == "CPU_PCIe_bandwidth_GBps")  { CPU_PCIe_bandwidth_GBps = std::stod(value); }
        else if (command == "CPU_memory_line_GB")       { CPU_memory_line_GB = std::stod(value); }
        // simulation GPU statistics
        else if (command == "GPU_PCIe_bandwidth_GBps")  { GPU_PCIe_bandwidth_GBps = std::stod(value); }
        else if (command == "GPU_memory_size_GB")       { GPU_memory_size_GB = std::stod(value); }
        else if (command == "GPU_frequency_GHz")        { GPU_frequency_GHz = std::stod(value); }
        else if (command == "GPU_malloc_uspB")          { GPU_malloc_uspB = std::stod(value); }
        else if (command == "GPU_free_uspB")            { GPU_free_uspB = std::stod(value); }
        // simulation SSD statistics
        else if (command == "SSD_PCIe_bandwidth_GBps")  { SSD_PCIe_bandwidth_GBps = std::stod(value); }
        else if (command == "SSD_read_latency_us")      { SSD_read_latency_us = std::stod(value); }
        else if (command == "SSD_write_latency_us")     { SSD_write_latency_us = std::stod(value); }
        else if (command == "SSD_latency_us")           { SSD_latency_us = std::stod(value); }
        // simulation PCIe statistics
        else if (command == "PCIe_latency_us")          { PCIe_latency_us = std::stod(value); }
        else if (command == "PCIe_batch_size_page")     { PCIe_batch_size_in_page = std::stoi(value); }
        // simulation Timing sentivity statistics
        else if (command == "kernel_time_std_dev")      { kernel_time_std_dev = std::stod(value); }
        else if (command == "ran_seed")                 { ran_seed = std::stoi(value); }
        else if (command == "kernel_speedup")           { kernel_speedup = std::stod(value); }
        // comments or empty line
        else if (command == "#" || command == "")       {}
        else {
          eprintf("Error: Invalid config entry <%s>, aborting...\n", command.c_str());
          Assert(false);
        }
    }
    // sanity check
    Assert((int) Simulator::GPUPageTable::EvcPolicy::DEEPUM != (int) Simulator::MigPolicy::DEEPUM);

    bool transform_input;
    if (nn_model_input_file.size() != 0) {
        Assert(orig_kernel_time_file.size() != 0);
        Assert(workspace_size_file.size() != 0);
        Assert(input_pf_kernel_time_file.size() != 0);
        Assert(pf_kernel_time_file.size() != 0);
        transform_input = true;
    } else {
        Assert(tensor_info_file.size() != 0);
        Assert(kernel_info_file.size() != 0);
        Assert(kernel_aux_time_file.size() != 0);
        transform_input = false;
    }

    if (transform_input) {
        // indirection if there is no file is fed through stdin
        if (isatty(fileno(stdin))) {
        if (nn_model_input_file.empty()) {
            eprintf("No input NN model in either stdin or config file\n", "");
        } else {
            // open a file and redirect to stdin
            std::ifstream nn_model(nn_model_input_file.c_str());
            if (!nn_model.good()) {
                eprintf("Invalid input NN model specified in config file <%s>\n",
                        nn_model_input_file.c_str());
                Assert(false);
            }
            if (is_transformer!=1) {
                freopen(nn_model_input_file.c_str(), "r", stdin);
            }
        }
        }
    }
    // parameter transformation
    if (output_folder_name.back() == '/') output_folder_name.pop_back();
    stat_output_file = output_folder_name + "/" + stat_output_file;

    if (is_simulation) {
        // eviction policy
        std::transform(eviction_policy_str.begin(), eviction_policy_str.end(), eviction_policy_str.begin(), ::toupper);
        if (eviction_policy_str == "RANDOM") {
            eviction_policy = Simulator::GPUPageTable::EvcPolicy::RANDOM;
        } else if (eviction_policy_str == "LRU" || eviction_policy_str == "TOLERANT") {
            eviction_policy = Simulator::GPUPageTable::EvcPolicy::LRU;
        } else if (eviction_policy_str == "GUIDED") {
            eviction_policy = Simulator::GPUPageTable::EvcPolicy::GUIDED;
        } else if (eviction_policy_str == "DEEPUM") {
            eviction_policy = Simulator::GPUPageTable::EvcPolicy::DEEPUM;
        } else {
            wprintf("Defaulting eviction policy to be LRU\n", "");
            eviction_policy = Simulator::GPUPageTable::EvcPolicy::LRU;
        }
        // migration policy
        std::transform(migration_policy_str.begin(), migration_policy_str.end(), migration_policy_str.begin(), ::toupper);
        if (migration_policy_str == "DEEPUM") {
            migration_policy = Simulator::MigPolicy::DEEPUM;
        } else {
            wprintf("Defaulting migration policy to be OURS\n", "");
            migration_policy = Simulator::MigPolicy::OURS;
        }
    }

    // parameter validation
    if (is_simulation) {
        SimulationParamSanityCheck();
    } else {
        if (is_input_pf_only) Assert(is_UVM);
    }
    // only one or less than one of these options are specified
    Assert(is_resnet + is_inception + is_senet <= 1);

    printf("End configs\n\n");

    // set random seed
    srand(0);

    bool output_folder_exists = system(("test -d " + output_folder_name).c_str()) == 0;
    if (output_folder_exists && !output_override) {
        wprintf("Output folder <%s> exists\n", output_folder_name.c_str());
    }

    // cout redirection
    RedirStdOut* r;

    ParseCommandLine(argc, argv);

    SetupOutputFolder();

    if (transform_input) {
        if (is_transformer == 1) {
            transformer_parse(nn_model_input_file.c_str());
            transformer_op_datalow_pass(borden);
        } else {
            InitScanner();
            InitParser();
            yyparse();
            layer_pre_pass_datasize();
            layer_first_pass_dataflow();
        }

        if (!is_simulation) {
            // tensor info
            r = new RedirStdOut("tensors.config");
            for (size_t i = 0; i < tensor_list.size(); i++) {
                tensor_list[i]->print();
            }
            delete r;
        }

        if (is_transformer == 1) {
            // layer info
            r = new RedirStdOut("layers.config");
            for (size_t i = 0; i < forward_ops.size(); i++) {
                forward_ops[i]->print();
            }
            delete r;
            transformer_scheduling_kernels();
        } else {
            // layer info
            r = new RedirStdOut("layers.config");
            for (size_t i = 0; i < forward_layers.size(); i++) {
                forward_layers[i]->print();
            }
            delete r;

            layer_second_pass_scheduling_kernels();
        }
    } else {
        loadKernelInfo();
    }

    if (is_simulation) {
        if (transform_input) {
            loadKernelTimes();
            loadWorkspaceSizes();
            std::string token;

            std::istringstream ss1(config_file_path);
            std::vector<std::string> vec1;
            while(std::getline(ss1, token, '/')) vec1.push_back(token);

            std::istringstream ss2(vec1[vec1.size() - 1]);
            std::vector<std::string> vec2;
            while(std::getline(ss2, token, '-')) vec2.push_back(token);

            string dnn_name = vec1[vec1.size() - 2];
            int batch_size = std::stoi(vec2[0]);
            exportKernelInfo(dnn_name, batch_size);
        }

        // tensor info
        r = new RedirStdOut("tensors.config");
        for (size_t i = 0; i < tensor_list.size(); i++) {
            tensor_list[i]->print();
        }
        delete r;

        // kernel info
        r = new RedirStdOut("kernels.config");
        for (size_t i = 0; i < kernel_list.size(); i++) {
            kernel_list[i].print();
        }
        delete r;

        nprintf("Global Memory amount:       %12lld B (%8.2f GB)\n", memory_offset_weights, memory_offset_weights / pow(1024, 3));
        nprintf("Intermediate Memory amount: %12lld B (%8.2f GB)\n", memory_offset_intermediate, memory_offset_intermediate / pow(1024, 3));
        nprintf("Memory Overcommitment:      %lld B/%lld B, %f GB/%f GB (%f%%)\n",
                memory_offset_intermediate + memory_offset_weights, (long long) (GPU_memory_size_GB * std::pow(1024, 3)),
                (memory_offset_intermediate + memory_offset_weights) / std::pow(1024, 3), GPU_memory_size_GB,
                (memory_offset_intermediate + memory_offset_weights) / (GPU_memory_size_GB * std::pow(1024, 3)) * 100);
        long max_num_pages = 0;
        CUDAKernel *max_mem_usage_kernel = nullptr;
        for (auto it = kernel_list.begin(); it != kernel_list.end(); ++it) {
            CUDAKernel *current_kernel = &(*it);
            vector<Tensor *> required_tensors;
            current_kernel->getRequiredTensors(required_tensors);
            long num_pages = 0;
            for (Tensor *tensor : required_tensors) {
                num_pages += std::ceil((float) tensor->size_in_byte / PAGE_SIZE);
            }
            if (num_pages > max_num_pages) {
                max_num_pages = num_pages;
                max_mem_usage_kernel = current_kernel;
            }
        }
        double max_memory_usage_GB = max_num_pages * PAGE_SIZE / std::pow(1024, 3);
        Assert(max_mem_usage_kernel != nullptr);
        nprintf("Memory Usage Maximized at Kernel%d: %lld B (%f GB)\n",
                max_mem_usage_kernel->kernel_id, max_num_pages * PAGE_SIZE,
                max_memory_usage_GB);
        if (max_memory_usage_GB > GPU_memory_size_GB) {
            eprintf("Single kernel memory usage %f GB greater than total GPU memory size %f GB, aborting",
                    max_memory_usage_GB, GPU_memory_size_GB);
            Assert(false);
        }

        if (transform_input) {
            exit(0);
        }

        tensor_first_pass_liveness_analysis();
        tensor_second_pass_interval_formation();
        get_interval_time();

        // life cycle info
        r = new RedirStdOut("interval.config");
        for (int i = 0; i < tensor_list.size(); i++) {
            tensor_list[i]->print_liveness();
            tensor_list[i]->print_intervals();
        }
        delete r;


        give_eviction_guide();

        // eviction guide
        // r = new RedirStdOut("evc_guide.config");
        // print_eviction_guide_table();
        // delete r;

        r = new RedirStdOut("pre_dealloc.config");
        scheduling_prefetch();
        delete r;

        // prefetch guide
        r = new RedirStdOut("prefetch_guide.config");
        print_prefetch_table();
        delete r;

        // real memory usage
        r = new RedirStdOut("real_mem.config");
        print_GPU_mem_really_in_use();
        delete r;

        // kernel time table
        r = new RedirStdOut("kernel_time_table.config");
        for (int i = 0; i < kernel_list.size(); i++) {
            std::cout << kernel_time_table[i] << std::endl;
        }
        delete r;

        nprintf("Average interval time: %f ms\n", interval_list[(interval_list.size() - 1) / 2]->time_estimated);
        iprintf("Checking output stat files\n", "");
        Simulator::Stat stat(stat_output_file);
        if (!stat.outputFileExists()) {
            if (kernel_time_std_dev != 0) {
                printf("Kernel time variation with std %f\n", kernel_time_std_dev);
                std::uniform_real_distribution<double> distribution(1 - kernel_time_std_dev, 1 + kernel_time_std_dev);
                if (ran_seed != 1)
                {
                    rand_device.seed((unsigned int)(ran_seed));
                }
                // rand_device.seed((unsigned int)(100*kernel_time_std_dev));
                for (int i = 0; i < kernel_list.size(); i++) {
                    double ratio = distribution(rand_device);
                    if (ratio < 0.1) ratio = 0.1; // prevent normal distribution to produce a negative number
                    if (ratio > 1.9) ratio = 1.9; // ensure that the mean is still around 1.0
                    kernel_list[i].execution_cycles *= ratio;
                    kernel_list[i].input_pf_execution_cycles *= ratio;
                    kernel_list[i].pf_execution_cycles *= ratio;
                    Assert(kernel_list[i].execution_cycles > 0);
                    Assert(kernel_list[i].input_pf_execution_cycles > 0);
                    Assert(kernel_list[i].pf_execution_cycles > 0);
                }
            }
            iprintf("\nPerforming Simulation\n", "");
            Simulator::EventSimulator *sim = new Simulator::EventSimulator(stat_output_file);
            sim->run(num_iteration);
            delete sim; // make sure stats are written back to the files
        }
        iprintf("\nPerforming Analysis\n", "");
        stat.prepareOutputFiles(true);
        stat.analyzeStat();
    } else {
        if (is_cudnn) {
            iprintf("Generating main code -- CUDNN mode\n", "");
            auto start_time = high_resolution_clock::now();
            // cudnn_profiling(true);          // normal run, individual
            cudnn_profiling(false);         // normal run, grouped
            // cudnn_profiling(false, true);   // workspace only
            duration<float> fsec = high_resolution_clock::now() - start_time;
            iprintf("Profiling duration: %fs (%fms)\n", fsec.count(), fsec.count() * 1000);
        } else {
            wprintf("Profiling without CUDNN deprecated\n", "");
            iprintf("Generating main code -- %s mode\n", is_individual ? "individual" : "whole");
            main_code_generation();
            printf("\n");

            if (is_compile || is_run) {
                printf("Profiling with Individual: %s, Compile: %s, Run: %s\n",
                    is_individual ? "True" : "False",
                    is_compile ? "True" : "False",
                    is_run ? "True" : "False");
                // run profiling scripts
                std::string args = " ";
                if (is_compile)
                    args += "-c ";
                if (is_compile && compile_max_thread_num > 0)
                    args += "-t " + to_string(compile_max_thread_num) + " ";
                if (is_run)
                    args += "-r ";
                Assert(system((output_folder_name + "/scripts/compileAndRun.sh" + args).c_str()) == 0);
            } else {
                iprintf("Both Compile and Run are disabled, run <%s> manually to profile\n",
                    (output_folder_name + "/scripts/compileAndRun.sh").c_str());
            }
        }
    }

    for (int i = 0; i < forward_layers.size(); i++)
    {
      delete forward_layers[i];
    }
    for (int i = 0; i < tensor_list.size(); i++)
    {
      delete tensor_list[i];
    }


    return (ReportError::NumErrors() == 0? 0 : -1);
}
