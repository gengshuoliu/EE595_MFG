

#ifndef INFERENCE_H
#define INFERENCE_H
#include <vector>
#include <string>
#include <torch/script.h>
using namespace std;


struct inference_thread_data {
    // Store data used by each thread.
    int thread_id;
    int num_row_per_thread;
    int total_row;
    torch::Tensor * input;
    at::Tensor * result;
    torch::jit::script::Module  * model;
};

class Inference {
    public:
        Inference(string & input_src);
        // Read model. Different child classes read different models.
        virtual void read_model() = 0;
        // Read input. Specified using argv[1]
        void read_input();
        // Do inference. num_thread specified using argv[2]. By default num_thread = 1.
        void forward(int num_thread);
        // Get the output for a single input.
        at::Tensor get_result_at(int idx);
        // output inference result out.
        friend ostream & operator << (ostream &out, const Inference &inf);
        // Save result at filename.
        void save_result(string filename);
    protected:
        at::Tensor result;
        torch::Tensor input;
        torch::jit::script::Module model;
        string input_src;
        // Function for a single thread.
        static void * forward_worker(void *arg);

};

// Do inference with FNO model.
// FNO model is serialized using the method described in https://pytorch.org/tutorials/advanced/cpp_export.html#step-1-converting-your-pytorch-model-to-torch-script.
// The serialized model should be stored as SerializedFNO.pt.
class InferenceWithFNO:public Inference {
    public:
        InferenceWithFNO(string & input_src);
        
        void read_model();
        
};

#endif