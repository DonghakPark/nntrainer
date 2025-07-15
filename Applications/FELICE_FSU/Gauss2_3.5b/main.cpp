#include <algorithm>
#include <array>
#include <cctype> // std::tolower
#include <chrono>
#include <ctime>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <acti_func.h>
#include <common.h>
#include <layer.h>
#include <model.h>
#include <optimizer.h>

#include "gauss.hpp"
#include <app_context.h>

////@note If you set MODEL_SIZE 3b, don't forget to set gqa_size = 2;
///////// If you set MODEL_SIZE=0.2b, please load weight
// 3b / gauss2.5
const std::string MODEL_SIZE = "3b"; // {"0.2b", "1b", "3.5b", "3b"}
const std::string MODEL_NAME = "Gauss2.5";
const std::string TASK = "summarization"; // {"summarization", "keyword_search"}
const std::string W_TYPE = "Q4_K";        // {"BCQ", "FP32", "FP16", "Q4_K"}
const std::string A_TYPE = "FP32";        // {"FP32", "FP16"}

int main(int argc, char *argv[]) {

  int DIM, NUM_LAYERS, NUM_HEADS, NUM_VOCAB, NUM_KEY_VALUE_HEADS, INIT_SEQ_LEN,
    MAX_SEQ_LEN, INTERMEDIATE_SIZE;
  std::set<std::string> lora_target;
  float NORM_EPS = 1e-6;
  std::string WEIGHT_HOME;
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0] << " 0 false 0 [weight_home]"
              << std::endl;
    return -1;
  }
  WEIGHT_HOME = argv[4];

  if (MODEL_SIZE == "0.2b") {
    DIM = 768;
    INTERMEDIATE_SIZE = 3072;
    NUM_LAYERS = 20;
    NUM_HEADS = 12;
    NUM_KEY_VALUE_HEADS = 12;
    NUM_VOCAB = 96000;
    INIT_SEQ_LEN = 1024;
    MAX_SEQ_LEN = 1024;
    lora_target = {};
  } else if (MODEL_SIZE == "1b") {
    DIM = 1536; // hidden_size
    INTERMEDIATE_SIZE = 6144;
    NUM_LAYERS = 28;         // num_hidden_layers
    NUM_HEADS = 12;          // num_attention_heads
    NUM_KEY_VALUE_HEADS = 2; // num_key_value_heads
    NUM_VOCAB = 105900;      // vocab_size
    NORM_EPS = 1e-5;         // rms_norm_eps
    INIT_SEQ_LEN = 1024;
    MAX_SEQ_LEN = 262144;
    if (TASK == "keyword_search")
      lora_target = {"layer0_wv",
                     "layer1_wv",
                     "layer2_wv",
                     "layer3_wv",
                     "layer4_wv",
                     "layer5_wv",
                     "layer6_wv",
                     "layer7_wv",
                     "layer8_wv",
                     "layer9_wv",
                     "layer10_wv",
                     "layer11_wv",
                     "layer12_wv",
                     "layer13_wv",
                     "layer14_wv",
                     "layer15_wv",
                     "layer16_wv",
                     "layer17_wv",
                     "layer18_wv",
                     "layer19_wv",
                     "layer20_wv",
                     "layer21_wv",
                     "layer22_wv",
                     "layer23_wv",
                     "layer24_wv",
                     "layer25_wv",
                     "layer26_wv",
                     "layer27_wv",

                     "layer0_wk",
                     "layer1_wk",
                     "layer2_wk",
                     "layer3_wk",
                     "layer4_wk",
                     "layer5_wk",
                     "layer6_wk",
                     "layer7_wk",
                     "layer8_wk",
                     "layer9_wk",
                     "layer10_wk",
                     "layer11_wk",
                     "layer12_wk",
                     "layer13_wk",
                     "layer14_wk",
                     "layer15_wk",
                     "layer16_wk",
                     "layer17_wk",
                     "layer18_wk",
                     "layer19_wk",
                     "layer20_wk",
                     "layer21_wk",
                     "layer22_wk",
                     "layer23_wk",
                     "layer24_wk",
                     "layer25_wk",
                     "layer26_wk",
                     "layer27_wk",

                     "layer0_attention_out",
                     "layer1_attention_out",
                     "layer2_attention_out",
                     "layer3_attention_out",
                     "layer4_attention_out",
                     "layer5_attention_out",
                     "layer6_attention_out",
                     "layer7_attention_out",
                     "layer8_attention_out",
                     "layer9_attention_out",
                     "layer10_attention_out",
                     "layer11_attention_out",
                     "layer12_attention_out",
                     "layer13_attention_out",
                     "layer14_attention_out",
                     "layer15_attention_out",
                     "layer16_attention_out",
                     "layer17_attention_out",
                     "layer18_attention_out",
                     "layer19_attention_out",
                     "layer20_attention_out",
                     "layer21_attention_out",
                     "layer22_attention_out",
                     "layer23_attention_out",
                     "layer24_attention_out",
                     "layer25_attention_out",
                     "layer26_attention_out",
                     "layer27_attention_out"};
    else
      lora_target = {};
  } else {
    DIM = 3072; // hidden_size
    INTERMEDIATE_SIZE = 8192;
    NUM_LAYERS = 34;         // num_hidden_layers
    NUM_HEADS = 24;          // num_attention_heads (Query Head #)
    NUM_KEY_VALUE_HEADS = 2; // num_key_value_heads (Key/Value Head #)
    NUM_VOCAB = 105900;      // vocab_size
    NORM_EPS = 1e-5;         // rms_norm_eps
    INIT_SEQ_LEN = 10;       // ???
    MAX_SEQ_LEN = 163840;    // max_position_embeddings

    if (TASK == "keyword_search")
      lora_target = {"layer0_wv",
                     "layer1_wv",
                     "layer2_wv",
                     "layer3_wv",
                     "layer4_wv",
                     "layer5_wv",
                     "layer6_wv",
                     "layer7_wv",
                     "layer8_wv",
                     "layer9_wv",
                     "layer10_wv",
                     "layer11_wv",
                     "layer12_wv",
                     "layer13_wv",
                     "layer14_wv",
                     "layer15_wv",
                     "layer16_wv",
                     "layer17_wv",
                     "layer18_wv",
                     "layer19_wv",
                     "layer20_wv",
                     "layer21_wv",
                     "layer22_wv",
                     "layer23_wv",
                     "layer24_wv",
                     "layer25_wv",
                     "layer26_wv",
                     "layer27_wv",
                     "layer28_wv",
                     "layer29_wv",
                     "layer30_wv",
                     "layer31_wv",
                     "layer32_wv",
                     "layer33_wv",

                     "layer0_wk",
                     "layer1_wk",
                     "layer2_wk",
                     "layer3_wk",
                     "layer4_wk",
                     "layer5_wk",
                     "layer6_wk",
                     "layer7_wk",
                     "layer8_wk",
                     "layer9_wk",
                     "layer10_wk",
                     "layer11_wk",
                     "layer12_wk",
                     "layer13_wk",
                     "layer14_wk",
                     "layer15_wk",
                     "layer16_wk",
                     "layer17_wk",
                     "layer18_wk",
                     "layer19_wk",
                     "layer20_wk",
                     "layer21_wk",
                     "layer22_wk",
                     "layer23_wk",
                     "layer24_wk",
                     "layer25_wk",
                     "layer26_wk",
                     "layer27_wk",
                     "layer28_wk",
                     "layer29_wk",
                     "layer30_wk",
                     "layer31_wk",
                     "layer32_wk",
                     "layer33_wk",

                     "layer1_wq",
                     "layer2_attention_out",
                     "layer6_attention_out",
                     "layer7_attention_out",
                     "layer9_attention_out",
                     "layer10_attention_out",
                     "layer11_attention_out",
                     "layer12_attention_out",
                     "layer13_attention_out",
                     "layer14_attention_out",
                     "layer16_attention_out",
                     "layer18_attention_out",
                     "layer19_attention_out",
                     "layer19_wq",
                     "layer20_attention_out",
                     "layer21_attention_out"};
    else
      lora_target = {};
  }

  bool SMART_REPLY = false;
  bool USE_VOCAB_SELECTION = false;
  bool FSU = true;
  int LSH_CHOICES = 512;
  int NUM_TO_GENERATE = 1024;
  int batch_size = 1;
  int epochs = 1;
  unsigned int LOCAL_WINDOW_SIZE = UINT_MAX;

  const std::string MODEL_TENSOR_TYPE = W_TYPE + "-" + A_TYPE;
  std::string WEIGHT_FILE =
    "nntr_gauss2.5_summarization_q4k_3b_q6k_tieword.bin";
  // "nntr_" + MODEL_NAME + "_" + TASK + "_" + W_TYPE +
  //                           "_" + MODEL_SIZE + ".bin";
  std::transform(WEIGHT_FILE.begin(), WEIGHT_FILE.end(), WEIGHT_FILE.begin(),
                 [](char c) { return std::tolower(c); });
  WEIGHT_FILE = WEIGHT_HOME + WEIGHT_FILE;

  std::cout << "[W-A Type] " << MODEL_TENSOR_TYPE << std::endl;
  std::cout << "[Weight File] " << WEIGHT_FILE << std::endl;

  try {
    // Gauss assumes 0 false 0 as a default.
    // Other options are not implemented yet
    const std::vector<std::string> args(argv + 1, argv + argc);
#ifdef _WIN32
    bool apply_temp = _stricmp("true", args[1].c_str()) == 0;
#else
    bool apply_temp = (strcasecmp("true", args[1].c_str()) == 0);
#endif
    unsigned int mode = std::stoi(args[2]);

    // Create Gauss
    // In the Gauss constructor, `createAndRun` is called
    Gauss gauss(INIT_SEQ_LEN, NUM_VOCAB, DIM, INTERMEDIATE_SIZE, SMART_REPLY, NUM_LAYERS,
                USE_VOCAB_SELECTION, LSH_CHOICES, MAX_SEQ_LEN, NUM_HEADS,
                NUM_KEY_VALUE_HEADS, NUM_TO_GENERATE, MODEL_TENSOR_TYPE,
                batch_size, epochs, FSU, NORM_EPS, "./tokenizer.json",
                LOCAL_WINDOW_SIZE, lora_target);
    // load weight
    gauss.load_weight(WEIGHT_FILE);
    // gauss.save_weight("without_tie_word");
    // Make an input for gauss
    if (TASK == "summarization") {
#if defined(_WIN32)
      std::wstring input_text =
        L""
#else
      std::string input_text =
        ""
#endif
        "<|begin_of_text|><|begin_of_text|><|turn_start|>System\n<|turn_end|>\n<|turn_start|>"
        "User\n"
        "You are a summarization expert.Please read the provided <Text> "
        "carefully and summarize it in 3 sentences in English."
        "The summary should comprehensively cover the entire content of the "
        "original text and be written with the same meaning as the source "
        "material."
        "<|begin_of_text|><|turn_start|>System<|turn_end|>"
        " YG엔터테인먼트 소속 그룹 아이콘의 멤버였던 가수 비아이(23·본명 "
        "김한빈)에 대한 부실수사 논란을 놓고 검경 이 책임공방을 벌이고 "
        "있다.CBS보도에 따르면 비아이에 대한 마약 제보를 받았던 당시 검찰은 "
        "가수 승리(본명 이승현·29)의 마약 투약 의혹 을 포착하고 수사를 벌이고 "
        "있었던 것으로 확인됐다.당시 검찰이 YG 소속가수들의 마약 의혹을 잇따라 "
        "포착했으면서도 별다른 처벌 없이 사건이 종결된 이유에 대 해 의구심이 "
        "증폭되고 있다.검찰 측은 YG 연예인의 마약 의혹에 집중하고 있던 당시 "
        "한서희 사건에 포함된 비아이 마약 투약 정황에 대해 살펴보지 않은 "
        "이유에 대해 비중 있는 연예인에게(수사의) 관심이 돌아가는 사항이었다며 "
        "비아이는 그렇게 비중  있는 연예인이 아니었다고 말한 것으로 "
        "전해졌다.법조계에 따르면, 당시 수원지검은 승리가 강남유명 클럽 "
        "아레나에서 엑스터시를 투약했다는 제보를 받은 뒤 승 리의 자택에서 간이 "
        "마약 검사를 하고, 소변과 모발 등 체모까지 제출받아 검사했다.다만 "
        "검사결과는 음성으로 나와 검찰은 승리를 불기소 처분한 것으로 "
        "알려졌다.이 시 기는 수원지검이 비아이가 언급된 한서희씨의 "
        "마약투약의혹을 경기 용인동부경찰서로부터 송치받아 조사 하던 때와 별반 "
        "차이가 없다.경찰은 검찰이 YG 관련 사건을 수사하려 하니 빨리 사건을 "
        "넘겨달라고 했다고 주장하고 검찰측은 그런 요구를 한 적이 없다고 팽팽히 "
        "맞섰다.수원지검은 그해 8월 경기 용인동부경찰서로부터 한서희 씨 마약 "
        "투약 사건을 넘겨받으면서 비아이 관련 보고서 도 함께 받았지만 그를 "
        "입건하거나 소환조사하지 않았다.검찰 관계자는 경찰이 비아이에 대해 "
        "내사를 진행하는  것으로 이해하고 우리는 별도로 수사를 진행하지 않았다 "
        "고 경찰 측에 책임을 떠넘겼다.하지만 검찰이 비슷한 시기 YG 소속이었던 "
        "승리의 마약 의혹을 포착한 것으로 확인되면서, 비아이건을 경찰로부 터 "
        "나름 상세히 보고받고도 추가로 확인하지 않은 부분에 의구심이 커지고 "
        "있다.검찰은 함께 입건된 다른 피의자들은 일사천리로 처리하면서, YG "
        "소속 가수 비아이의 마약 혐의를 진술한 제보 자 한서희의 사건만 "
        "방치했다.앞서 KBS는 검찰이 공익제보자가 진술했던 YG 소속 가수 "
        "비아이의 마약 혐의도 직 접 확인하지 않았고 내사중이 라던 경찰에도 "
        "물어보지 않았다고 보도했다.YG 마약 의혹을 밝힐 핵심 증인으로 지목해 "
        "놓고선 출국 금지 조치도 하지 않아 한서희는 2016년 12월 9일 미 국으로 "
        "출국했고 열흘 뒤 검찰은 그에 대해 기소 중지를 내렸다.검찰 관계자는 "
        "당시 제보자의 변호인이 해외 공연이 있어 두 달 동안 미국에서 "
        "체류한다고 말해, 기소 중지할 수 밖에 없었다고 밝혔다.하지만 한서희는 "
        "연예인이 아닌 일반인이 락 공연일정 등이 있을 리가 없었다.이런 말로 "
        "기소중지를 받게 한 변 호사는 YG측이 선임해 준 변호사였다고 "
        "전해진다.국민권익위원회는 양현석 전 YG 대표가 개입해 비아이 마약 "
        "사건을 덮으려 했다는 공익 신고 사건을 대검찰청 에 넘겼다. "
        "\n<|turn_end|>\n<|turn_start|>Assistant\n";
#if defined(_WIN32)
      wchar_t output_text[1024];
      gauss.run(input_text.c_str(), apply_temp, mode, output_text, 1024);
#else
      std::string output_text;
      gauss.run(input_text, apply_temp, mode, output_text, 1024);
#endif
    } else if (TASK == "keyword_search") {
#if defined(_WIN32)
      std::wstring input_text =
        L""
#else
      std::string input_text =
        ""
#endif
        "<|begin_of_text|><|begin_of_text|><|turn_start|>System\n<|turn_end|>"
        "\n<|turn_start|>"
        "User\n[{<LOCALE>}]\nko-KR\n아래 [{<INPUT>}]에서 검색에 필요한 "
        "Key-Value "
        "값을 JSON 형태로 출력해줘.\n[{<INPUT>}]\nPDF 파일 중에서 2022년 7월에 "
        "작성된 것 보여줘<|turn_end|>\n<|turn_start|>Assistant\n";

#if defined(_WIN32)
      wchar_t output_text[1024];
      gauss.run(input_text.c_str(), apply_temp, mode, output_text, 1024);
#else
      std::string output_text;
      gauss.run(input_text, apply_temp, mode, output_text, 1024);
#endif
    }

  } catch (const std::exception &e) {
    std::cerr << "uncaught error while running! details: " << e.what()
              << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
