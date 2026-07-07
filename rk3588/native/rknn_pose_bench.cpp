#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "rknn_api.h"

static std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("failed to open " + path);
  }
  const auto size = file.tellg();
  std::vector<uint8_t> data(static_cast<size_t>(size));
  file.seekg(0, std::ios::beg);
  file.read(reinterpret_cast<char*>(data.data()), size);
  return data;
}

static void check(int ret, const char* what) {
  if (ret != RKNN_SUCC) {
    std::cerr << what << " failed: " << ret << std::endl;
    std::exit(1);
  }
}

static const char* type_name(rknn_tensor_type type) {
  switch (type) {
    case RKNN_TENSOR_FLOAT32: return "float32";
    case RKNN_TENSOR_FLOAT16: return "float16";
    case RKNN_TENSOR_INT8: return "int8";
    case RKNN_TENSOR_UINT8: return "uint8";
    case RKNN_TENSOR_INT16: return "int16";
    case RKNN_TENSOR_UINT16: return "uint16";
    case RKNN_TENSOR_INT32: return "int32";
    case RKNN_TENSOR_UINT32: return "uint32";
    default: return "unknown";
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " model.rknn [loops=300] [warmup=20] [want_float=1]" << std::endl;
    return 2;
  }

  const std::string model_path = argv[1];
  const int loops = argc > 2 ? std::atoi(argv[2]) : 300;
  const int warmup = argc > 3 ? std::atoi(argv[3]) : 20;
  const int want_float = argc > 4 ? std::atoi(argv[4]) : 1;

  auto model = read_file(model_path);
  rknn_context ctx = 0;
  check(rknn_init(&ctx, model.data(), static_cast<uint32_t>(model.size()), 0, nullptr), "rknn_init");
  check(rknn_set_core_mask(ctx, RKNN_NPU_CORE_0_1_2), "rknn_set_core_mask");

  rknn_input_output_num io_num;
  std::memset(&io_num, 0, sizeof(io_num));
  check(rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)), "query io num");
  std::cout << "inputs=" << io_num.n_input << " outputs=" << io_num.n_output << std::endl;

  rknn_tensor_attr in_attr;
  std::memset(&in_attr, 0, sizeof(in_attr));
  in_attr.index = 0;
  check(rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr)), "query input attr");
  std::cout << "input name=" << in_attr.name << " dims=[";
  for (uint32_t i = 0; i < in_attr.n_dims; ++i) {
    std::cout << in_attr.dims[i] << (i + 1 == in_attr.n_dims ? "" : ",");
  }
  std::cout << "] fmt=" << get_format_string(in_attr.fmt) << " type=" << type_name(in_attr.type)
            << " size=" << in_attr.size << std::endl;

  std::vector<rknn_tensor_attr> out_attrs(io_num.n_output);
  for (uint32_t i = 0; i < io_num.n_output; ++i) {
    std::memset(&out_attrs[i], 0, sizeof(out_attrs[i]));
    out_attrs[i].index = i;
    check(rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &out_attrs[i], sizeof(out_attrs[i])), "query output attr");
    std::cout << "output" << i << " name=" << out_attrs[i].name << " dims=[";
    for (uint32_t d = 0; d < out_attrs[i].n_dims; ++d) {
      std::cout << out_attrs[i].dims[d] << (d + 1 == out_attrs[i].n_dims ? "" : ",");
    }
    std::cout << "] fmt=" << get_format_string(out_attrs[i].fmt) << " type=" << type_name(out_attrs[i].type)
              << " size=" << out_attrs[i].size << std::endl;
  }

  std::vector<uint8_t> input_buf(640 * 640 * 3, 114);
  rknn_input input;
  std::memset(&input, 0, sizeof(input));
  input.index = 0;
  input.buf = input_buf.data();
  input.size = static_cast<uint32_t>(input_buf.size());
  input.pass_through = 0;
  input.type = RKNN_TENSOR_UINT8;
  input.fmt = RKNN_TENSOR_NHWC;

  std::vector<rknn_output> outputs(io_num.n_output);
  std::vector<double> times_ms;
  times_ms.reserve(loops);

  for (int i = 0; i < warmup + loops; ++i) {
    check(rknn_inputs_set(ctx, 1, &input), "inputs_set");
    auto t0 = std::chrono::steady_clock::now();
    check(rknn_run(ctx, nullptr), "rknn_run");
    std::memset(outputs.data(), 0, sizeof(rknn_output) * outputs.size());
    for (auto& out : outputs) {
      out.want_float = static_cast<uint8_t>(want_float ? 1 : 0);
      out.is_prealloc = 0;
    }
    check(rknn_outputs_get(ctx, io_num.n_output, outputs.data(), nullptr), "outputs_get");
    auto t1 = std::chrono::steady_clock::now();
    check(rknn_outputs_release(ctx, io_num.n_output, outputs.data()), "outputs_release");

    if (i >= warmup) {
      times_ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
  }

  const double total_ms = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
  double min_ms = times_ms.empty() ? 0.0 : times_ms[0];
  double max_ms = min_ms;
  for (double v : times_ms) {
    if (v < min_ms) min_ms = v;
    if (v > max_ms) max_ms = v;
  }
  const double avg_ms = total_ms / static_cast<double>(times_ms.size());
  std::cout << "loops=" << loops << " warmup=" << warmup << " want_float=" << want_float
            << " avg_ms=" << avg_ms << " min_ms=" << min_ms << " max_ms=" << max_ms
            << " fps=" << (1000.0 / avg_ms) << std::endl;

  rknn_destroy(ctx);
  return 0;
}
