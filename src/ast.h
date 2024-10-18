/* File: ast.h
 * ----------- 
 * This file defines the abstract base class Node and the concrete 
 * Identifier and Error node subclasses that are used through the tree as 
 * leaf nodes. A parse tree is a hierarchical collection of ast nodes (or, 
 * more correctly, of instances of concrete subclassses such as VarDecl,
 * ForStmt, and AssignExpr).
 * 
 * Location: Each node maintains its lexical location (line and columns in 
 * file), that location can be NULL for those nodes that don't care/use 
 * locations. The location is typcially set by the node constructor.  The 
 * location is used to provide the context when reporting semantic errors.
 *
 * Parent: Each node has a pointer to its parent. For a Program node, the 
 * parent is NULL, for all other nodes it is the pointer to the node one level
 * up in the parse tree.  The parent is not set in the constructor (during a 
 * bottom-up parse we don't know the parent at the time of construction) but 
 * instead we wait until assigning the children into the parent node and then 
 * set up links in both directions. The parent link is typically not used 
 * during parsing, but is more important in later phases.
 *
 * Printing: The only interesting behavior of the node classes for pp2 is the 
 * bility to print the tree using an in-order walk.  Each node class is 
 * responsible for printing itself/children by overriding the virtual 
 * PrintChildren() and GetPrintNameForNode() methods. All the classes we 
 * provide already implement these methods, so your job is to construct the
 * nodes and wire them up during parsing. Once that's done, printing is a snap!

 */

#ifndef _H_ast
#define _H_ast

#include <stdlib.h>   // for NULL
#include <string>
#include <vector>
#include <assert.h>



typedef enum {
  Conv2d_T, ReLU_T, MaxPool2d_T, AdaptiveAvgPool2d_T, Linear_T, Dropout_T, BatchNorm2d_T, Init_T, Add_T, Concat_T, Scale_T
} OperatorType;




class Hidding_Interval;

class Tensor
{
    private:
        Tensor();
    public:
        Tensor(long long size, bool glob = false);
        unsigned long getGlobalOffset();
        std::string name() const;
        bool is_alive(int current_kernel) const;
        void print() const;
        void print_liveness();
        void print_intervals();

        int tensor_id;
        long long size_in_byte;
        long long raw_size_byte;
        long long address_offset;
        bool is_global_weight;
        bool is_choosed_to_evict = false;
        int live_interval[2]; //live_interval[0] = birth; live_interval[1] = death; if death=-1, it means that this tensor is always dead
        std::vector<Hidding_Interval*> hidding_intervals;

    //Flashneuron only: (starts with 'f')
        bool f_is_allocated_on_GPU = false;
        bool f_is_choosed_to_offload = false;
        bool f_is_fetching = false;
        long f_page_range[2];
};

class Hidding_Interval
{
  public:
    double time_estimated;   //us
    int kernelLevel_interval[2];
    int original_prefetch_index;
    int evict_finish_index;
    bool is_looped;
    bool is_offloaded;
    bool is_really_offloaded;
    long GPU_mem_line;
    Tensor* the_tensor;
    Hidding_Interval(Tensor* t, long GPU_line){the_tensor = t; is_looped = false; is_offloaded = false; is_really_offloaded =false; GPU_mem_line = GPU_line; original_prefetch_index = -1; evict_finish_index = -1;};
    void print();
};



#endif
