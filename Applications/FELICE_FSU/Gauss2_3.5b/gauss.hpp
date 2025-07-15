#ifndef __GAUSS_HPP__
#define __GAUSS_HPP__ __GAUSS_HPP__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#define WSTR std::wstring
#define WCHAR_P wchar_t *
#else
#define WIN_EXPORT
#define WSTR std::string
#define WCHAR_P std::string &
#endif

#include <memory>
#include <stdexcept>
#include <vector>

#include <common.h>
#include <layer.h>
#include <model.h>
#include <stdlib.h>

#include <app_context.h>              // appcontext
#include <custom_fc_layer.h>          // custom fc layer
#include <custom_fc_lora_layer.h>     // custom fc_lora_layer
#include <custom_mha_core_layer.h>    // custom mha core layer
#include <custom_mha_core_v2_layer.h> // custom mha core layer
#include <custom_qkv_layer.h>
#include <custom_rms_norm.h> // custom rms_norm
#include <custom_swiglu.h>   // custom swiglu
#include <custom_tie_word_embedding_layer.h>
#include <engine.h>
#include <llm_util.hpp> // llm utils
#include <set>
#include <tokenizers_c.h>
#include <tokenizers_cpp.h>

/***************** ALIAS *******************/
using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;
using ml::train::createLayer;

/***************** GAUSS ******************/
/**
 * @class GAUSS wrapping class.
 * @brief  Gauss class, used in testing Gauss refactoring
 * @note Gauss class interanaly contains model. Any methods required for
 * Gauss construction is implemented as private methods.
 */
WIN_EXPORT class Gauss {

public:
  /**
   * @brief Construct a new Gauss object
   * In the constructor, it saves key hyperparameters for Gauss,
   * and create layers to be used in the Gauss
   */
  WIN_EXPORT Gauss(unsigned int INIT_SEQ_LEN = 1024,
                   int const NUM_VOCAB = 96000, int const DIM = 768,
                   int const INTERMEDIATE_SIZE = 4 * 768,
                   bool SMART_REPLY = false, int const NUM_LAYERS = 20,
                   bool USE_VOCAB_SELECTION = false, int LSH_CHOICES = 512,
                   unsigned int MAX_SEQ_LEN = 1024, int NUM_HEADS = 12,
                   int NUM_KEY_VALUE_HEADS = 12, int NUM_TO_GENERATE = 512,
                   std::string MODEL_TENSOR_TYPE = "BCQ-FP32",
                   int batch_size = 1, int epochs = 1, bool FSU = false,
                   float NORM_EPS = 0.00001,
                   const char *tokenizer_file = nullptr,
                   unsigned int LOCAL_WINDOW_SIZE = 1024,
                   const std::set<std::string> &lora_target = {});
  WIN_EXPORT ~Gauss() { free(ids_history); }

  /**
   * @brief load weight of Gauss model
   */
  WIN_EXPORT void load_weight(const std::string &weight_path);

  /**
   * @brief load weight of Gauss model
   */
  WIN_EXPORT void save_weight(const std::string &weight_path);

  /**
   * @brief run Gauss inference with text input
   */
  WIN_EXPORT void run(WSTR text_, bool apply_temperature, unsigned int mode);

  WIN_EXPORT void run(const WCHAR_P text_, bool apply_temperature,
                      unsigned int mode, WCHAR_P output_ptr,
                      size_t output_size);

  WIN_EXPORT void reRun(const WCHAR_P text_, bool apply_temperature,
                        unsigned int mode, WCHAR_P output_ptr,
                        size_t output_size, unsigned int prev_idx,
                        const char *kvcache_path_);

  WIN_EXPORT void save_kvcache(std::string path);

  WIN_EXPORT void load_kvcache(std::string path);

  WIN_EXPORT void pause(unsigned int &idx);

  WIN_EXPORT void setLoraPath(std::string path);

  WIN_EXPORT void setLora(bool enable);

  // WIN_EXPORT void saveLora(std::string path);

  // WIN_EXPORT void loadLora(bool enable);

  /**
   * @brief generate token from logits
   */
  WIN_EXPORT std::vector<unsigned int> generate(
    float *logits, unsigned int NUM_VOCAB = 0, unsigned int NUM_BATCH = 1,
    bool do_sample = false, float temperature = 1, unsigned int top_k = 1,
    float top_p = 1, float repetition_penalty = 1,
    unsigned int *input_ids = nullptr, unsigned int NUM_INPUT_IDS = 0,
    unsigned int *bad_words_ids = nullptr, unsigned int NUM_BAD_WORDS_IDS = 0);

private:
  /**
   * @brief construct Model
   */
  void constructModel(const std::set<std::string> &lora_target = {});

  /**
   * @brief create decoder
   */
  std::vector<LayerHandle>
  createTransformerDecoder(const int layer_id, std::string input_name,
                           const std::set<std::string> &lora_target = {});

  /**
   * @brief create attention layer, which is called by createTransformerDecoder
   */
  std::vector<LayerHandle>
  createAttentionLayer(const int layer_id, int seq_len, int n_heads,
                       int head_dim, std::string query_name,
                       std::string key_name, std::string value_name,
                       const std::set<std::string> &lora_target = {});

  /**
   * @brief create FFN layer, which is called by createTransformerDecoder
   */
  std::vector<LayerHandle> createFeedForwardLayer(const int layer_id, int dim,
                                                  int hidden_dim,
                                                  std::string input_name);

  /**
   * @brief decode IDs to text and store it in the corresponding batch
   *
   * @param tokenizer GPT2Encoder to decode ids
   * @param ids list of id
   */
  void registerOutputs(std::unique_ptr<tokenizers::Tokenizer> &tokenizer,
                       std::vector<unsigned int> ids, unsigned int pos,
                       const std::vector<bool> &eos_list);

  /** Gauss model */
  ModelHandle model;

  /** key hyper-parameters of Gauss structure */
  unsigned int INIT_SEQ_LEN;
  int const NUM_VOCAB;
  int const DIM;
  int const INTERMEDIATE_SIZE;
  bool SMART_REPLY;
  int const NUM_LAYERS;
  bool USE_VOCAB_SELECTION;
  int LSH_CHOICES;
  unsigned int MAX_SEQ_LEN;
  int NUM_HEADS;
  int NUM_KEY_VALUE_HEADS;
  int NUM_TO_GENERATE;
  std::string MODEL_TENSOR_TYPE;
  unsigned int LOCAL_WINDOW_SIZE;

  /**  hyper-parameters fr Gauss training */
  unsigned int batch_size;
  int epoch;

  /** constant */
  int const MULTIPLE_OF = 256;

  int MODE = 0;
  unsigned int *BAD_WORD_IDS;
  unsigned int NUM_BADWORDS;
  bool MEMORY_SWAP;
  unsigned int gqa_size;
  float NORM_EPS = 0.00001; // rms_norm_eps

  /** tokenizer-related path */
  std::string tokenizer_file_name;

  /** internal buffer */
  unsigned int *ids_history;
  std::vector<std::string> output_list;

  std::string lora_path;
  std::string instruct_path;

  float *prev_input_sample;

  unsigned int token_generation_idx;
  bool is_pause = false;
  bool is_prefill_finished = false;

  std::mt19937 rng; // random number gen.
};

#endif // gauss_hpp
