#ifndef EXECUTOR_H_
#define EXECUTOR_H_

#include <vector>
#include <unordered_set>
#include "context.h"
#include "node.h"

class Executor {
public:
  // ctx is the context the executor run, either cpu or gpu
  // out is the output node
  // node_need_grads are the grads we need to take derivate wrt
  Executor(const Context& ctx, 
           const Node& out,
           const std::vector<Node>& node_need_grads) 
        : ctx_(ctx), out_(out), node_need_grads_(node_need_grads) {
    // Use auto diff to complete the graph
    Gradient();
    need_topo_order_ = true;
  }

  void Run(const std::vector<Node>& out_nodes, 
           std::vector<Tensor>& out_vals, 
           const std::vector<Node>& grad_nodes,
           std::vector<Tensor>& grad_vals,
           std::unordered_map<Node, Tensor>& node_to_tensor) {
    // nodes represent the nodes we need to evalute, 
    // so we need to do topo sort with nodes as output first.
    std::vector<Node> nodes;
    nodes.insert(nodes.end(), out_nodes.begin(), out_nodes.end());
    std::transform(grad_nodes.begin(), grad_nodes.end(), 
                   std::back_inserter(nodes), 
                   [this](const Node& node) { return node_to_grads_.at(node); });

    if (need_topo_order_) GetTopoOrder(nodes);

    for (auto node : topo_orders_) {
      std::vector<Node> in_nodes;
      node.GetInputNodes(in_nodes);
      if (in_nodes.size() == 0) continue;
      std::vector<Tensor> in_tensors;
      std::vector<TensorShape> in_shapes;
      for (auto in_node : in_nodes) {
        const Tensor& in_tensor = node_to_tensor[in_node];
        in_tensors.push_back(in_tensor);
        in_shapes.push_back(in_tensor.GetTensorShape());
      }


      std::vector<TensorShape> out_shapes;
      node.GetOp()->Infer(node, in_shapes, out_shapes);

      std::vector<Tensor> out_tensors = {Tensor(out_shapes[0], Context::cpu())};
      node.GetOp()->Compute(node, in_tensors, out_tensors);

      node_to_tensor[node] = out_tensors[0];
    }

    out_vals.clear();
    for (auto node : out_nodes) {
      out_vals.push_back(node_to_tensor[node]);
    }
    grad_vals.clear();
    for (auto node : grad_nodes) {
      grad_vals.push_back(node_to_tensor[node_to_grads_[node]]);
    }
  }


private:
  void Gradient() {
    // A map for node -> grads
    std::unordered_map<Node, std::vector<Node>> node_to_grads;

    auto reduce_sum_by_node = [&node_to_grads] (const Node& node) {
      auto grads = node_to_grads.at(node);
      if (grads.size() > 1) {
        for (int i = 1; i < grads.size(); i++) {
          grads[0] += grads[i];
        }
      }
      return grads[0];
    };

    GetTopoOrder({out_});
    node_to_grads[out_].push_back(OnesOperator(out_));
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

    for (auto node : node_need_grads_) {
      node_to_grads_[node] = reduce_sum_by_node(node);
    }
  }

  void GetTopoOrder(const std::vector<Node>& outs) {
    need_topo_order_ = false;
    topo_orders_.clear();
    std::unordered_set<Node> visited;
    for (auto node : outs) {
      dfs(node, visited);
    }
  }

  void dfs(const Node& node, std::unordered_set<Node>& visited) {
    if (visited.find(node) == visited.end()) {
      if (!node.IsVariable()) {
        std::vector<Node> input_nodes;
        node.GetInputNodes(input_nodes);
        for (auto input_node : input_nodes) {
          dfs(input_node, visited);      
        } 
      }
      visited.insert(node);
      topo_orders_.push_back(node);
    } 
  };

  Context ctx_;
  Node out_;
  std::vector<Node> node_need_grads_;
  std::unordered_map<Node, Node> node_to_grads_; 
  std::vector<Node> topo_orders_;
  bool need_topo_order_;
};

#endif 
