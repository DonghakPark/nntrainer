// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   gauss.cpp
 * @brief  Wrapping class for gauss (refactored from gauss.cpp)
 * @date   21 August 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <codecvt>
#include <fcntl.h>
#include <gauss.hpp>
#include <tokenizers_cpp.h>

/**
 * @brief constructor of gauss
 */
Gauss::Gauss(unsigned int INIT_SEQ_LEN, int const NUM_VOCAB, int const DIM,
             int const INTERMEDIATE_SIZE, bool SMART_REPLY,
             int const NUM_LAYERS, bool USE_VOCAB_SELECTION, int LSH_CHOICES,
             unsigned int MAX_SEQ_LEN, int NUM_HEADS, int NUM_KEY_VALUE_HEADS,
             int NUM_TO_GENERATE, std::string MODEL_TENSOR_TYPE, int batch_size,
             int epochs, bool fsu, float NORM_EPS, const char *tokenizer_file,
             unsigned int LOCAL_WINDOW_SIZE,
             const std::set<std::string> &lora_target) :
  INIT_SEQ_LEN(INIT_SEQ_LEN),
  NUM_VOCAB(NUM_VOCAB),
  DIM(DIM),
  INTERMEDIATE_SIZE(INTERMEDIATE_SIZE),
  SMART_REPLY(SMART_REPLY),
  NUM_LAYERS(NUM_LAYERS),
  USE_VOCAB_SELECTION(USE_VOCAB_SELECTION),
  LSH_CHOICES(LSH_CHOICES),
  MAX_SEQ_LEN(MAX_SEQ_LEN),
  NUM_HEADS(NUM_HEADS),
  NUM_KEY_VALUE_HEADS(NUM_KEY_VALUE_HEADS),
  NUM_TO_GENERATE(NUM_TO_GENERATE),
  MODEL_TENSOR_TYPE(MODEL_TENSOR_TYPE),
  batch_size(batch_size),
  epoch(epochs),
  BAD_WORD_IDS{},
  NUM_BADWORDS(0),
  MEMORY_SWAP(fsu),
  gqa_size(NUM_HEADS / NUM_KEY_VALUE_HEADS),
  NORM_EPS(NORM_EPS),
  tokenizer_file_name(tokenizer_file),
  LOCAL_WINDOW_SIZE(LOCAL_WINDOW_SIZE) {

  /** initialize */
  for (int b = 0; b < batch_size; ++b)
    output_list.push_back("");

  /** register custom layers used in gauss */
  auto &ct_engine = nntrainer::Engine::Global();
  auto app_context =
    static_cast<nntrainer::AppContext *>(ct_engine.getRegisteredContext("cpu"));

  try {
    app_context->registerFactory(
      nntrainer::createLayer<custom::CustomSwiGLULayer>);
    app_context->registerFactory(
      nntrainer::createLayer<custom::CustomRMSNormLayer>);
    app_context->registerFactory(nntrainer::createLayer<custom::CustomFCLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<custom::CustomMHACoreV2Layer>);
    app_context->registerFactory(
      nntrainer::createLayer<custom::CustomTieWordEmbeddingLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<custom::FullyConnectedLayer>);
    app_context->registerFactory(nntrainer::createLayer<custom::QKVLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
  }
  // construct gauss model

  constructModel(lora_target);

  // setup model
  model->setProperty({withKey("batch_size", batch_size),
                      withKey("epochs", epochs),
                      withKey("model_tensor_type", MODEL_TENSOR_TYPE),
                      withKey("save_path", "test_model.bin"),
                      // withKey("fsu", MEMORY_SWAP ? "true" : "false"),
                      // withKey("fsu_lookahead", "28")
  });

  // setting optimizer
  auto optimizer = ml::train::createOptimizer("sgd", {"learning_rate=0.001"});
  model->setOptimizer(std::move(optimizer));

  // compile model
  int status = model->compile(ml::train::ExecutionMode::INFERENCE);
  if (status) {
    throw std::invalid_argument("model compilation failed!");
  }

  // initialize model
  if (model->initialize(ml::train::ExecutionMode::INFERENCE)) {
    throw std::invalid_argument("model initialization failed!");
  }

  // allocate memory for the internal buffer
  ids_history =
    (unsigned int *)malloc(batch_size * MAX_SEQ_LEN * sizeof(unsigned int));
};

void Gauss::load_weight(const std::string &weight_path) {
  instruct_path = weight_path;

  // load weight
  try {
    model->load(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
  } catch (const std::exception &e) {
    std::cerr << "Error during loadFromWeight:" << e.what() << std::endl;
  }

  //  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);
};

void Gauss::save_weight(const std::string &weight_path) {

  // load weight
  try {
    model->save(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
  } catch (const std::exception &e) {
    std::cerr << "Error during saveFromWeight:" << e.what() << std::endl;
  }
};

void Gauss::constructModel(const std::set<std::string> &lora_target) {

  // layers used in the model
  std::vector<LayerHandle> layers;

  // create model
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  // create input layer
  layers.push_back(createLayer(
    "input", {withKey("name", "input0"),
              withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));

  // create embedding layer

  layers.push_back(
    createLayer("custom_tie_word_embedding",
                {"name=embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
                 "weight_dtype=Q6_K", "out_dim=" + std::to_string(DIM),
                 SMART_REPLY ? "smart_reply=true" : "smart_reply=false"}));

  // create transformer layers
  for (int i = 0; i < NUM_LAYERS; ++i) {
    std::vector<LayerHandle> transformer;
    if (i == 0)
      transformer = createTransformerDecoder(0, "embedding0", lora_target);
    else
      transformer = createTransformerDecoder(
        i, "layer" + std::to_string(i - 1) + "_decoder_output", lora_target);
    layers.insert(layers.end(), transformer.begin(), transformer.end());
  }

  // create rms_norm
  layers.push_back(createLayer(
    "custom_rms_norm",
    {withKey("name", "output_norm"),
     withKey("epsilon", std::to_string(NORM_EPS)),
     withKey("input_layers",
             "layer" + std::to_string(NUM_LAYERS - 1) + "_decoder_output"),
     withKey("packed", "false"),
     withKey("smart_reply", SMART_REPLY ? "true" : "false")}));

  // create lm_head layer
  //
  // layers.push_back(createLayer(
  //   "custom_tie_word_embedding",
  //   {withKey("name", "output_of_gauss"), withKey("unit", NUM_VOCAB),
  //    withKey("disable_bias", "true"), withKey("input_layers", "output_norm"),
  //    withKey("weight_dtype", "Q6_K"),
  //    withKey("use_vocab_selection", USE_VOCAB_SELECTION ? "true" : "false"),
  //    withKey("lsh_choices", LSH_CHOICES), withKey("shared_from",
  //    "embedding0"), withKey("smart_reply", SMART_REPLY ? "true" :
  //    "false")}));

  // add created layers into the model
  for (auto &layer : layers) {
    model->addLayer(layer);
  }
}

/**
 * @brief create transformer decoder layers
 */
std::vector<LayerHandle>
Gauss::createTransformerDecoder(const int layer_id, std::string input_name,
                                const std::set<std::string> &lora_target) {

  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "custom_rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
     withKey("input_layers", input_name),
     withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false"),
     withKey("smart_reply", SMART_REPLY ? "true" : "false")}));

  auto att_layer = createAttentionLayer(
    layer_id, INIT_SEQ_LEN, NUM_HEADS, DIM / NUM_HEADS,
    "layer" + std::to_string(layer_id) + "_attention_norm",
    "layer" + std::to_string(layer_id) + "_attention_norm",
    "layer" + std::to_string(layer_id) + "_attention_norm", lora_target);

  layers.insert(layers.end(), att_layer.begin(), att_layer.end());

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_add"),
     withKey("input_layers", input_name + ",layer" + std::to_string(layer_id) +
                               "_attention_out")}));

  layers.push_back(createLayer(
    "custom_rms_norm",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_decoder_add"),
     withKey("epsilon", std::to_string(NORM_EPS)), withKey("packed", "false"),
     withKey("smart_reply", SMART_REPLY ? "true" : "false")}));

  auto ffn_layer =
    createFeedForwardLayer(layer_id, DIM, INTERMEDIATE_SIZE,
                           "layer" + std::to_string(layer_id) + "_ffn_norm");
  layers.insert(layers.end(), ffn_layer.begin(), ffn_layer.end());

  layers.push_back(createLayer(
    "addition",
    {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output"),
     withKey("input_layers", "layer" + std::to_string(layer_id) +
                               "_decoder_add,layer" + std::to_string(layer_id) +
                               "_ffn_output")}));

  return layers;
};

/**
 * @brief create Attention Layer for the seperate implementation
 */
std::vector<LayerHandle>
Gauss::createAttentionLayer(const int layer_id, int seq_len, int n_heads,
                            int head_dim, std::string query_name,
                            std::string key_name, std::string value_name,
                            const std::set<std::string> &lora_target) {

  std::vector<LayerHandle> layers;

  auto Q = "layer" + std::to_string(layer_id) + "_wq";
  auto K = "layer" + std::to_string(layer_id) + "_wk";
  auto V = "layer" + std::to_string(layer_id) + "_wv";
  auto A = "layer" + std::to_string(layer_id) + "_attention";
  auto O = "layer" + std::to_string(layer_id) + "_attention_out";
  auto QKV = "layer" + std::to_string(layer_id) + "_qkv";
  auto QKV0 = QKV + "(0)";
  auto QKV1 = QKV + "(1)";
  auto QKV2 = QKV + "(2)";

  // Helper lambda to check and add LoRA parameters
  auto add_lora_params = [&](std::vector<std::string> &params,
                             const std::string &name) {
    if (lora_target.find(name) != lora_target.end()) {
      // params.emplace_back(withKey("lora_rank", 32));
      // params.emplace_back(withKey("lora_alpha", 128));
      // params.emplace_back(withKey("tensor_dtype", {"Q4_K", "FP32"}));
      params.emplace_back(withKey("lora_enable", "true"));
    }
  };

  // V layer
  std::vector<std::string> v_params = {
    withKey("name", V), withKey("unit", head_dim * n_heads / gqa_size),
    withKey("disable_bias", "true"), withKey("input_layers", value_name),
    withKey("weight_initializer", "ones")};
  add_lora_params(v_params, V);
  layers.push_back(createLayer("custom_fc_lora", v_params));

  // K layer
  std::vector<std::string> k_params = {
    withKey("name", K), withKey("unit", head_dim * n_heads / gqa_size),
    withKey("disable_bias", "true"), withKey("input_layers", key_name),
    withKey("weight_initializer", "ones")};
  add_lora_params(k_params, K);
  layers.push_back(createLayer("custom_fc_lora", k_params));

  // Q layer
  std::vector<std::string> q_params = {
    withKey("name", Q), withKey("unit", head_dim * n_heads),
    withKey("disable_bias", "true"), withKey("input_layers", query_name),
    withKey("weight_initializer", "ones")};
  add_lora_params(q_params, Q);
  layers.push_back(createLayer("custom_fc_lora", q_params));

  // Attention core layer
  std::vector<std::string> a_params = {
    withKey("name", A),
    withKey("num_heads", n_heads),
    withKey("num_heads_kv", n_heads / gqa_size),
    withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)),
    withKey("local_window_size",
            (layer_id + 1) % 5 ? LOCAL_WINDOW_SIZE : UINT_MAX),
    withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)),
    withKey("input_layers", {Q, K, V}),
    withKey("smart_reply", SMART_REPLY ? "true" : "false")};
  add_lora_params(a_params, A);
  layers.push_back(createLayer("custom_mha_core_v2", a_params));

  // O layer
  std::vector<std::string> o_params = {
    withKey("name", O), withKey("unit", head_dim * n_heads),
    withKey("disable_bias", "true"), withKey("input_layers", A),
    withKey("weight_initializer", "ones")};
  add_lora_params(o_params, O);
  layers.push_back(createLayer("custom_fc_lora", o_params));

  return layers;
}

std::vector<LayerHandle> Gauss::createFeedForwardLayer(const int layer_id,
                                                       int dim, int hidden_dim,
                                                       std::string input_name) {

  std::vector<LayerHandle> layers;

  layers.push_back(
    createLayer("custom_fc_lora",
                {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_1"),
                 withKey("unit", hidden_dim), withKey("disable_bias", "true"),
                 withKey("input_layers", input_name),
                 withKey("weight_initializer", "ones")}));
  layers.push_back(
    createLayer("custom_fc_lora",
                {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_2"),
                 withKey("unit", hidden_dim), withKey("disable_bias", "true"),
                 withKey("input_layers", input_name),
                 withKey("weight_initializer", "ones")}));

  layers.push_back(createLayer(
    "custom_swiglu",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
     withKey("input_layers", "layer" + std::to_string(layer_id) + "_ffn_1," +
                               "layer" + std::to_string(layer_id) + "_ffn_2"),
     withKey("smart_reply", SMART_REPLY ? "true" : "false")}));

  layers.push_back(createLayer(
    "custom_fc_lora",
    {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_output"),
     withKey("unit", dim), withKey("disable_bias", "true"),
     withKey("input_layers",
             "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
     withKey("weight_initializer", "ones")}));

  return layers;
};
void PrintEncodeResult(const std::vector<int> &ids) {
  std::cout << "tokens=[";
  for (size_t i = 0; i < ids.size(); ++i) {
    if (i != 0)
      std::cout << ", ";
    std::cout << ids[i];
  }
  std::cout << "]" << std::endl;
}
std::string LoadBytesFromFile(const std::string &path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    std::cerr << "Cannot open " << path << std::endl;
    exit(1);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

void Gauss::run(WSTR text_, bool apply_temperature, unsigned int mode) {
  output_list.clear();
  auto json = LoadBytesFromFile(tokenizer_file_name);
  auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(json);
  for (unsigned int b = 0; b < batch_size; ++b)
    output_list.push_back("");
  is_pause = false;
  is_prefill_finished = false;

#if defined(_WIN32)
  setlocale(LC_ALL, "");
#endif

  /** invalid case check */
  if (MAX_SEQ_LEN < INIT_SEQ_LEN)
    throw std::invalid_argument(
      "MAX_SEQ_LEN should be greater thatn INIT_SEQ_LEN");

  /** prepare tokenizer */
  std::vector<float *> input;
  std::vector<float *> label;

  // auto tokenizer = unwrap(GPT2Encoder::load(vocab_file_name,
  // merge_file_name),
  //                         "Error initializing GPT2 tokenizer \n");
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;

  // MODE == 0
  if (!MODE) {
#if defined(_WIN32)
    std::wcout << L"" << text_ << std::endl;
    // auto _input = tokenizer.encode(text_);
    auto _input = tokenizer->Encode(converter.to_bytes(text_));
#else
    // print input text
    std::cout << text_ << std::endl;
    // std::wstring text = converter.from_bytes(text_);
    auto _input = tokenizer->Encode(text_);
#endif

    unsigned int _len = _input.size();

    unsigned int num_allow_str = MAX_SEQ_LEN - NUM_TO_GENERATE;

    // | <------- MAX_SEQ_LEN -------> |
    //               ||
    // |<-- input -->||<-- generate -->|

    // prefill input ! input text from user
    // and system token to command to model is included
    std::vector<int64_t> init_input;

    unsigned text_len = _len;

    if (_len > num_allow_str)
      text_len = num_allow_str;

    // feed only available length
    // if _input is allowed, it feeds all of the _input
    // otherwise, feeds only a part of _input
    for (unsigned int i = 0; i < text_len; ++i)
      init_input.push_back(_input[i]);

    _input.clear();

    // make real input of the gauss
    unsigned int init_len = init_input.size();
    float *input_sample =
      (float *)malloc(sizeof(float) * batch_size * MAX_SEQ_LEN);

    std::vector<bool> eos_list(batch_size, false);

    unsigned int input_len = init_len;
    // unsigned int input_len =
    //   (init_len > INIT_SEQ_LEN) ? INIT_SEQ_LEN : init_len;

    token_generation_idx = input_len + 1;

    for (unsigned int b = 0; b < batch_size; ++b) {
      for (unsigned int i = 0; i < input_len; ++i) {
        input_sample[b * MAX_SEQ_LEN + i] = static_cast<float>(init_input[i]);
        ids_history[b * MAX_SEQ_LEN + i] = init_input[i];
      }
    }
    // start to prefil !!
    std::vector<int64_t> token_ids;

    input.push_back(input_sample);

    std::vector<ml::train::TensorDim> input_dims;
    ml::train::TensorDim input_dim(1, 1, input_len, DIM);
    input_dims.push_back(input_dim);

    model->resetInputDimension(input_dims);

    auto start_prefill =
      std::chrono::high_resolution_clock::now(); // log the start_prefill time
    auto output = model->incremental_inference(batch_size, input, label,
                                               input_len, 0, input_len, false);
    //////////////////////////////////////////////////////////////////////////////////////////
    nntrainer::TensorDim embedding_tensor_dim(
      {1, 1, static_cast<unsigned long long>(NUM_VOCAB),
       static_cast<unsigned long long>(DIM)});
    embedding_tensor_dim.setDataType(ml::train::TensorDim::DataType::Q6_K);
    nntrainer::Tensor embedding_tensor(embedding_tensor_dim, true);

    // get data from file
    std::ifstream weight_file(
      "./nntr_gauss2.5_summarization_q4k_3b_q6k_tieword.bin", std::ios::binary);
    embedding_tensor.read(weight_file, 0, true);

    nntrainer::TensorDim input_dim2(
      {1, 1, 1, static_cast<unsigned long long>(DIM)});
    nntrainer::TensorDim output_dim(
      {1, 1, 1, static_cast<unsigned long long>(NUM_VOCAB)});
    nntrainer::Tensor input_(input_dim2, true);
    nntrainer::Tensor hidden_(output_dim, true);

    for (unsigned int w = 0; w < DIM; ++w) {
      input_.setValue(0, 0, 0, w, output[0][w]);
    }

    input_.dot(embedding_tensor, hidden_);
    auto hidden_output = hidden_.getData();

    //////////////////////////////////////////////////////////////////////////////////////////
    // post process output of the model!
    // generate_multi_tokens
    std::vector<unsigned int> id_list(generate_multi_tokens(
      hidden_output, NUM_VOCAB, batch_size, 1, ids_history, _len));

    //
    if (init_len < INIT_SEQ_LEN)
      registerOutputs(tokenizer, id_list, init_len, eos_list);

    is_prefill_finished = true;
    // finish prefill
    auto finish_prefill = std::chrono::high_resolution_clock::now();
    auto prefill_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(finish_prefill -
                                                            start_prefill);

    /** Token Generation */
    for (unsigned int b = 0; b < batch_size; ++b)
      input_sample[b * MAX_SEQ_LEN] = static_cast<float>(id_list[b]);

    // start to generate
    unsigned int generation_cnt = 0;
    int64_t total_generation_duration = 0;

    int cnt_exit = 0;

    for (token_generation_idx = input_len + 1;
         token_generation_idx < input_len + 1 + NUM_TO_GENERATE;
         ++token_generation_idx) {

      auto start_generation = std::chrono::high_resolution_clock::now();

      auto output_interval = model->incremental_inference(
        batch_size, input, label, input_len, token_generation_idx - 1,
        token_generation_idx);

      for (unsigned int w = 0; w < DIM; ++w) {
        input_.setValue(0, 0, 0, w, output_interval[0][w]);
      }

      input_.dot(embedding_tensor, hidden_);
      hidden_output = hidden_.getData();

      std::vector<unsigned int> ids_list(
        generate(hidden_output, NUM_VOCAB, batch_size, false, 1, 1, 1, 1,
                 ids_history, _len, BAD_WORD_IDS, NUM_BADWORDS));

      if (token_generation_idx < input_len) {
        for (unsigned int b = 0; b < batch_size; ++b) {
          input_sample[b * MAX_SEQ_LEN] =
            static_cast<float>(init_input[token_generation_idx]);
        }
        registerOutputs(tokenizer, ids_list, token_generation_idx, eos_list);
      } else {
        for (unsigned int b = 0; b < batch_size; ++b) {
          input_sample[b * MAX_SEQ_LEN] = static_cast<float>(ids_list[b]);
        }
        registerOutputs(tokenizer, ids_list, token_generation_idx, eos_list);
      }

      auto finish_generation = std::chrono::high_resolution_clock::now();
      std::chrono::milliseconds generation_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(
          finish_generation - start_generation);
      total_generation_duration += generation_duration.count();
      ++generation_cnt;

      for (unsigned int j = 0; j < batch_size; ++j) {
        if (!eos_list[j] && (ids_list[j] == 37)) { // turn_end
          eos_list[j] = true;
        }
      }

      bool is_finish = true;
      for (unsigned int j = 0; j < batch_size; ++j) {
        if (!eos_list[j]) {
          is_finish = false;
          break;
        }
      }

      if (is_finish) {
        free(input_sample);
        break;
      }

      if (is_pause) {
        prev_input_sample = input_sample;
        break;
      }
      cnt_exit++;
      if (cnt_exit >= 1) {
        break;
      }
    }

    std::cout << "\n\n";
    std::cout << "prefill: " << init_len << " tokens, "
              << prefill_duration.count() << " ms, "
              << ((double)init_len / prefill_duration.count() * 1000)
              << " TPS\n";
    std::cout << "generation: " << generation_cnt << " tokens, "
              << total_generation_duration << " ms, "
              << ((double)generation_cnt / total_generation_duration * 1000)
              << " TPS\n";
  }
}

#if defined(_WIN32)
std::wstring utf8_to_wstring(const std::string &str) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> conv;
  return conv.from_bytes(str);
}
#endif

void Gauss::reRun(const WCHAR_P text_, bool apply_temperature,
                  unsigned int mode, WCHAR_P output_ptr, size_t output_size,
                  unsigned int prev_idx, const char *kvcache_path_) {
  auto json = LoadBytesFromFile(tokenizer_file_name);
  auto tokenizer = tokenizers::Tokenizer::FromBlobJSON(json);

  //  output_list.clear();
  //  for (unsigned int b = 0; b < batch_size; ++b)
  // output_list.push_back("");
  is_pause = false;

#if defined(_WIN32)
  setlocale(LC_ALL, "");
#endif

  /** invalid case check */
  if (MAX_SEQ_LEN < INIT_SEQ_LEN)
    throw std::invalid_argument(
      "MAX_SEQ_LEN should be greater thatn INIT_SEQ_LEN");

  /** prepare tokenizer */
  std::vector<float *> input;
  std::vector<float *> label;

  // auto tokenizer = unwrap(GPT2Encoder::load(vocab_file_name,
  // merge_file_name),
  //                         "Error initializing GPT2 tokenizer \n");

  // auto tokenizer = qualla::HFTokenizer(tokenizer_json_path);

  // MODE == 0
  if (!MODE) {
#if defined(_WIN32)
    std::wcout << L"" << text_ << std::endl;
    std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
    auto _input = tokenizer->Encode(converter.to_bytes(text_));
#else
    // print input text
    std::cout << text_ << std::endl;
    // std::wstring text = converter.from_bytes(text_);
    // std::wstring kvcache_path = converter.from_bytes(kvcache_path_);
    auto _input = tokenizer->Encode(text_);
#endif

    unsigned int _len = _input.size();

    unsigned int num_allow_str = MAX_SEQ_LEN - NUM_TO_GENERATE;

    // | <-------------- MAX_SEQ_LEN --------------> |
    //                      ||
    // |<-- input -->||<-- post -->||<-- generate -->|

    // prefill input ! input text from user
    // and system token to command to model is included
    std::vector<int64_t> init_input;

    unsigned text_len = _len;

    if (_len > num_allow_str)
      text_len = num_allow_str;

    // feed only available length
    // if _input is allowed, it feeds all of the _input
    // otherwise, feeds only a part of _input
    for (unsigned int i = 0; i < text_len; ++i)
      init_input.push_back(_input[i]);

    _input.clear();

    // make real input of the gauss
    unsigned int init_len = init_input.size();

    std::vector<bool> eos_list(batch_size, false);

    unsigned int input_len =
      (init_len > INIT_SEQ_LEN) ? INIT_SEQ_LEN : init_len;

    // // start to prefil !
    std::vector<int64_t> token_ids;

    float *input_sample = prev_input_sample;
    input.push_back(input_sample);

    // std::string cache_path(kvcache_path.begin(), kvcache_path.end());
    load_kvcache(kvcache_path_);

    // start to generate
    unsigned int generation_cnt = 0;
    int64_t total_generation_duration = 0;

    for (token_generation_idx = prev_idx;
         token_generation_idx < input_len + 1 + NUM_TO_GENERATE;
         ++token_generation_idx) {

      auto start_generation = std::chrono::high_resolution_clock::now();
      auto output_interval = model->incremental_inference(
        batch_size, input, label, MAX_SEQ_LEN - prev_idx,
        token_generation_idx - 1, token_generation_idx);

      std::vector<unsigned int> ids_list(
        generate(output_interval[0], NUM_VOCAB, batch_size, false, 1, 1, 1, 1,
                 ids_history, _len, BAD_WORD_IDS, NUM_BADWORDS));

      if (token_generation_idx < input_len) {
        for (unsigned int b = 0; b < batch_size; ++b) {
          input_sample[b * INIT_SEQ_LEN] =
            static_cast<float>(init_input[token_generation_idx]);
        }
        registerOutputs(tokenizer, ids_list, token_generation_idx, eos_list);

      } else {

        for (unsigned int b = 0; b < batch_size; ++b) {
          input_sample[b * INIT_SEQ_LEN] = static_cast<float>(ids_list[b]);
        }

        registerOutputs(tokenizer, ids_list, token_generation_idx, eos_list);
      }

      auto finish_generation = std::chrono::high_resolution_clock::now();
      std::chrono::milliseconds generation_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(
          finish_generation - start_generation);
      total_generation_duration += generation_duration.count();
      ++generation_cnt;

      for (unsigned int j = 0; j < batch_size; ++j) {
        if (!eos_list[j] &&
            (ids_list[j] == 37 || ids_list[j] == 0)) { // end of text
          eos_list[j] = true;
        }
      }

      bool is_finish = true;
      for (unsigned int j = 0; j < batch_size; ++j) {
        if (!eos_list[j]) {
          is_finish = false;
          break;
        }
      }
      if (is_finish) {
        is_prefill_finished = false;
        break;
      }
    }

    std::cout << "generation: " << generation_cnt << " tokens, "
              << total_generation_duration << " ms, "
              << ((double)generation_cnt / total_generation_duration * 1000)
              << " TPS\n";
    free(input_sample);
  }

#if defined(_WIN32)
  size_t offset = 0;
  for (const auto &s : output_list) {
    std::wstring w = utf8_to_wstring(s);
    size_t len = w.size();
    if (offset + len + 1 >= output_size)
      break;
    wcsncpy_s(output_ptr + offset, output_size - offset, w.c_str(), _TRUNCATE);
    offset += len + 1;
  }
#else
  for (const auto &str : output_list) {
    output_ptr += str;
  }
#endif
}

void Gauss::run(const WCHAR_P text_, bool apply_temperature, unsigned int mode,
                WCHAR_P output_ptr, size_t output_size) {

#if defined(_WIN32)
  std::wstring in(text_);
  run(in, apply_temperature, mode);
  size_t offset = 0;
  for (const auto &s : output_list) {
    std::wstring w = utf8_to_wstring(s);
    size_t len = w.size();
    if (offset + len + 1 >= output_size)
      break;
    wcsncpy_s(output_ptr + offset, output_size - offset, w.c_str(), _TRUNCATE);
    offset += len + 1;
  }
#else
  run(text_, apply_temperature, mode);
  for (const auto &str : output_list) {
    output_ptr += str;
  }
#endif
}

void Gauss::registerOutputs(std::unique_ptr<tokenizers::Tokenizer> &tokenizer,
                            std::vector<unsigned int> ids, unsigned int pos,
                            const std::vector<bool> &eos_list) {
  for (unsigned int i = 0; i < ids.size(); ++i) {
    if (!eos_list[i]) {
      auto decoded_str = tokenizer->Decode({static_cast<int>(ids[i])});
#if defined(_WIN32)
      std::wcout << L"" << utf8_to_wstring(decoded_str);
      std::wcout.flush();
#else
      std::cout << decoded_str;
      std::cout.flush();
#endif
      output_list[i] += decoded_str;
      ids_history[i * MAX_SEQ_LEN + pos] = ids[i];
    }
  }
}

std::vector<unsigned int>
Gauss::generate(float *logits, unsigned int NUM_VOCAB, unsigned int NUM_BATCH,
                bool do_sample, float temperature, unsigned int top_k,
                float top_p, float repetition_penalty, unsigned int *input_ids,
                unsigned int NUM_INPUT_IDS, unsigned int *bad_words_ids,
                unsigned int NUM_BAD_WORDS_IDS) {

  std::vector<unsigned int> outputs;
  for (unsigned int iteration = 0; iteration < NUM_BATCH; ++iteration) {

    // apply repetition penalty
    if (repetition_penalty != 1 && input_ids != nullptr && NUM_INPUT_IDS != 0) {
      applyRepetitionPenalty(logits, input_ids, NUM_INPUT_IDS,
                             repetition_penalty);
    }

    // apply bad words penalty
    if (bad_words_ids != nullptr && NUM_BAD_WORDS_IDS != 0) {
      applyBadWordsPenalty(logits, bad_words_ids, NUM_BAD_WORDS_IDS);
    }

    // return argmax if do_sample is false
    if (do_sample == false) {
      unsigned int argmax_idx =
        std::distance(logits, std::max_element(logits, logits + NUM_VOCAB));
      outputs.push_back(argmax_idx);
    } else {
      // apply temperature & top-k & top-p to logits
      float max_logits = applyTKP(logits, NUM_VOCAB, temperature, top_k, top_p);
      // transform logits to softmax
      float sum_exp_logits = 0;
      for (unsigned int i = 0; i < NUM_VOCAB; i++) {
        float exp_x = exp(logits[i] - max_logits);
        sum_exp_logits += exp_x;
        logits[i] = exp_x;
      }

      for (unsigned int i = 0; i < NUM_VOCAB; ++i) {
        logits[i] /= sum_exp_logits;
      }

      // sample from final logits
      std::discrete_distribution<int> dist(logits, logits + NUM_VOCAB);
      unsigned int sampled_idx = dist(rng);

      // add sampled word
      outputs.push_back(sampled_idx);
    }

    // set batch offset
    logits = logits + NUM_VOCAB;
    input_ids = input_ids + MAX_SEQ_LEN;
  }

  return outputs;
}

bool endsWithLoraEnable(const std::string &str) {
  const std::string suffix = "_loraenable";
  return str.size() >= suffix.size() &&
         str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

void Gauss::setLoraPath(std::string path) { lora_path = path; }

// void Gauss::saveLora(std::string path) {
//   model->save(path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
// }

void Gauss::setLora(bool enable) {
  if (lora_path.empty()) {
    std::cout << "lora_path empty" << std::endl;
    throw std::invalid_argument("lora_path is not set.");
  }

  std::ifstream f;
  if (enable)
    f = std::ifstream(lora_path, std::ios::binary);
  else
    f = std::ifstream(instruct_path, std::ios::binary);

  auto offset = 0;

  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [enable, &f, &offset](ml::train::Layer &l,
                               nntrainer::RunLayerContext &context, void *idx) {
      auto lora_enable = l.getProperty("lora_enable");
      if (l.getType() == custom::FullyConnectedLayer::type &&
          lora_enable != "empty" && lora_enable != std::to_string(enable)) {

        // l.setProperty({withKey("lora_enable", enable ? "true" : "false")});

        auto weight = context.getWeights().at(0);
        if (!enable) {
          auto prev_offset = weight->getVariableRef().getFileOffset();
          weight->getVariableRef().read(f, prev_offset, true);
        } else {
          weight->getVariableRef().read(f, offset, true);
          size_t size = weight->getVariable().getMemoryBytes();
          auto tensor_data_type = weight->getDim().getDataType();

          if (tensor_data_type != nntrainer::TensorDim::DataType::FP32 &&
              tensor_data_type != nntrainer::TensorDim::DataType::FP16 &&
              tensor_data_type != nntrainer::TensorDim::DataType::Q6_K) {
            // for tensor with qparam
            size += sizeof(uint16_t);
          }
          offset += size;
        }
      }
    };

  model->forEachLayer(fn, nullptr);
}

void Gauss::save_kvcache(std::string path) {

  auto f = nntrainer::checkedOpenStream<std::ofstream>(
    path, std::ios::out | std::ios::binary | std::ios::trunc);

  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [&f](ml::train::Layer &l, nntrainer::RunLayerContext &context,
              void *idx) {
      if (l.getType() == custom::CustomMHACoreV2Layer::type) {
        auto k_cache = context.getTensor(0);
        auto v_cache = context.getTensor(1);
        k_cache.save(f);
        v_cache.save(f);
      }
    };

  model->forEachLayer(fn, nullptr);
  f.close();
}

void Gauss::load_kvcache(std::string path) {
  auto f = nntrainer::checkedOpenStream<std::ifstream>(
    path, std::ios::in | std::ios::binary);

  std::function<void(ml::train::Layer &, nntrainer::RunLayerContext &, void *)>
    fn = [&f](ml::train::Layer &l, nntrainer::RunLayerContext &context,
              void *idx) {
      if (l.getType() == custom::CustomMHACoreV2Layer::type) {
        auto k_cache = context.getTensor(0);
        auto v_cache = context.getTensor(1);
        k_cache.read(f);
        v_cache.read(f);
      }
    };

  model->forEachLayer(fn, nullptr);
  f.close();
}

void Gauss::pause(unsigned int &idx) {
  while (!is_prefill_finished) {
    continue;
  }
  is_prefill_finished = true;
  is_pause = true;
  idx = token_generation_idx + 1;
}
