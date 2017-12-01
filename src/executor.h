#ifndef EXECUTOR_H_
#define EXECUTOR_H_

#include <vector>
#include <unordered_set>
#include "context.h"
#include "node.h"

class Executor {
public:
  Executor(const std::vector<Node>& outs, 
           const Context& ctx) 
      : outs_(outs), ctx_(ctx) {
    GetTopoOrder();
  }

  void Run(std::unordered_map<Node, Tensor>& node_to_tensor) {
    for (auto node : topo_orders_) {
      if (node_to_tensor.find(node) != node_to_tensor.end()) continue;

      std::vector<Node> input_nodes;
      node.GetInputNodes(input_nodes);

      std::vector<Tensor> input_tensors;
      std::vector<TensorShape> input_shapes;
      for (auto input_node : input_nodes) {
        const Tensor& input_tensor = node_to_tensor[input_node];
        input_tensors.push_back(input_tensor);
        input_shapes.push_back(input_tensor.GetTensorShape());
      }

      std::vector<TensorShape> out_shapes;
      node.GetOp()->Infer(input_shapes, out_shapes);
      std::vector<Tensor> out_tensors = {Tensor(out_shapes[0], Context::cpu())};

      node.GetOp()->Compute(input_tensors, out_tensors);
      node_to_tensor[node] = out_tensors[0];
    }
  }

  void Gradient(const Node& output_node, 
                const std::vector<Node>& inputs, 
                std::vector<Node>& outputs) {
    // A map for node -> grads
    std::unordered_map<Node, std::vector<Node>> node_to_grads;

    auto reduce_sum_by_node = [&node_to_grads] (const Node& node) {
      auto iter = node_to_grads.find(node);
      // TODO if we not find node
      auto grads = iter->second;
      if (grads.size() > 1) {
        for (int i = 1; i < grads.size(); i++) {
          grads[0] += grads[i];
        }
      }
      return grads[0];
    };

    node_to_grads[output_node].push_back(OnesOperator(output_node));
    for (auto iter = topo_orders_.rbegin(); iter != topo_orders_.rend(); iter++) {
      Node in_grad = reduce_sum_by_node(*iter);
      std::vector<Node> out_grads;
      if (iter->GetOp() != nullptr)
        iter->GetOp()->Gradient(*iter, in_grad, out_grads);
      std::vector<Node> inputs;
      iter->GetInputNodes(inputs);
      for (int i = 0; i < inputs.size(); i++) {
        node_to_grads[inputs[i]].push_back(out_grads[i]);
      }
    }

    for (auto node : inputs) {
      outputs.push_back(reduce_sum_by_node(node));
    }
  }

private:
  void GetTopoOrder() {
    std::unordered_set<Node> visited;
    for (auto node : outs_) {
      dfs(node, visited);
    }
  }

  void dfs(const Node& node, std::unordered_set<Node>& visited) {
    if (visited.find(node) == visited.end()) {
      std::vector<Node> input_nodes;
      node.GetInputNodes(input_nodes);
      if (!node.IsVariable()) {
        for (auto input_node : input_nodes) {
          dfs(input_node, visited);      
        } 
      }
      visited.insert(node);
      topo_orders_.push_back(node);
    } 
  };

  std::vector<Node> outs_;
  Context ctx_;
  std::vector<Node> topo_orders_;
};

#endif 
