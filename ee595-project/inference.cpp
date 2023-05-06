#include "inference.h"
#include <iostream>
#include <torch/torch.h>
Inference::Inference(string & input_src) {
    this->input_src = input_src;
}


void InferenceWithFNO::read_model() {
    this->model = torch::jit::load("SerializedFNO.pt");
    #ifdef DEBUG
    cout << "model read successfully\n";
    #endif
}

// Function invoked by each thread. Feed 1/num_thread data into the model.
void * Inference::forward_worker(void *arg) {
    struct inference_thread_data * thread_data = (struct inference_thread_data *) arg;
    int end = (thread_data->thread_id + 1) * thread_data->num_row_per_thread > thread_data->total_row ? thread_data->total_row : (thread_data->thread_id + 1) * thread_data->num_row_per_thread;
    int start = thread_data->thread_id * thread_data->num_row_per_thread;
    torch::Tensor * input = thread_data->input;
    #ifdef DEBUG
        cout << "size of thread " << thread_data->thread_id << " input inside forward_worker:\n";
        cout << "size: " << input->sizes()[0] << " * " << input->sizes()[1] << " * " << input->sizes()[2]  << " \n";
        cout << "start: " << start << " end: " << end << endl;
        cout << "size at 3: " <<input->index({2}).sizes()[0] << " * " << input->index({2}).sizes()[1] << " \n";
    #endif

    for (int i = start; i < end; i++) {
        std::vector<torch::jit::IValue> ivalue_input;
        ivalue_input.push_back(input->index({i}).reshape({1,256,3}));
        #ifdef DEBUG
        cout << "here " <<endl;
        cout << i << " " <<input->sizes()[0] << " " << input->sizes()[1] <<endl;
        #endif
        at::Tensor cur_result = thread_data->model->forward(ivalue_input).toTensor();
        #ifdef DEBUG
        cout << "here2 " <<endl;
        cout << i << " " <<input->sizes()[0] << " " << input->sizes()[1] <<endl;
        #endif
        thread_data->result->index({i}) = cur_result.reshape({256,3});

        #ifdef DEBUG
        cout << "here3 " <<endl;
        cout << i << " " <<input->sizes()[0] << " " << input->sizes()[1] <<endl;
        #endif
        
    }
    pthread_exit(NULL);
}


// Inference function. Create num_thread threads to feed the data.
void Inference::forward(int num_thread) {
    struct inference_thread_data thread_data[num_thread];
    pthread_t threads[num_thread];
    int rc;
    int num_row_per_thread = this->input.sizes()[0] / num_thread + (this->input.sizes()[0] % num_thread != 0);
    this->result = torch::empty({this->input.sizes()[0], 256, 3});
    #ifdef DEBUG
        cout << "result dim: " << this->result.sizes()[0] << endl;
    #endif
    for (int i = 0; i < num_thread ; i++) {
        thread_data[i].num_row_per_thread = num_row_per_thread;
        thread_data[i].total_row = this->input.sizes()[0];
        thread_data[i].thread_id = i;
        thread_data[i].input = &(this->input);
        thread_data[i].result = &(this->result);
        thread_data[i].model = &(this->model);
        #ifdef DEBUG
        cout << "thread data created successfully\n";
        cout << "size at 3: " << this->input[3].sizes()[0] << " * " << this->input[3].sizes()[1] << " \n";
        #endif
        rc = pthread_create(&threads[i], NULL, this->forward_worker, (void * ) (&thread_data[i]));
        if (rc) {
            fprintf(stderr, "pthread create error: %s\n", strerror(errno)); 
			exit(-1);
        }
    }

    for (int i = 0; i < num_thread; i++) {
        rc = pthread_join(threads[i], NULL);
        if (rc != 0){
            fprintf(stderr, "pthread joining error: %s\n", strerror(errno));
        }
        else {
            #ifdef DEBUG
                cout << "thread " << thread_data[i].thread_id << " joined\n";
            #endif
        }
    }
} 

// Get the result at index idx.
at::Tensor Inference::get_result_at(int idx) {
    return this->result.slice(0, idx, idx + 1);
}

// Output the result of type at::Tensor to out.
ostream & operator << (ostream &out, const Inference &inf) {
    out << inf.result;
    return out;
}

// Save result as filename.
void Inference::save_result(string filename) {
    torch::save(this->result, filename);
}

// Read input from input_src file.
void Inference::read_input() {
    torch::load(this->input, this->input_src);
    #ifdef DEBUG
    cout << "input loaded successfully ";
    cout << "size: " << this->input.sizes()[0] << " * " << this->input.sizes()[1] << " * " << this->input.sizes()[2] << " \n";
    cout << "size at 3: " << this->input[3].sizes()[0] << " * " << this->input[3].sizes()[1] << " \n";
    #endif

}

InferenceWithFNO::InferenceWithFNO(string & input_src) : Inference(input_src){}

int main(int argc, char *argv[]) {
    #ifdef DEBUG
    cout << "Debug mode\n\n";
    #endif
    string data_src = argv[1];
    InferenceWithFNO inf(data_src);
    inf.read_input();
    inf.read_model();
    int num_thread = stoi(argv[2]);
    if (argc < 3) {
        num_thread = 1;
    }
    inf.forward(num_thread);
    
    inf.save_result("result.pt");


    
}

