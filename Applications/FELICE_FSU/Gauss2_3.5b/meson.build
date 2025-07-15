# srllm_src = [
# 	  meson.current_source_dir() / 'main.cpp',
#   	  meson.current_source_dir() / 'gauss.cpp',
# 	  meson.current_source_dir() / 'llm_util.cpp',
#     custom_dot_wrapper_src_abs,
#     custom_rms_norm_src_abs,
#     custom_swiglu_src_abs,
#     custom_mha_src_abs,
#     custom_fc_src_abs,
#     custom_embedding_src_abs,
#     custom_lm_head_src_abs,
#     custom_fc_lora_src_abs,
#     custom_mha_core_v2_src_abs,
#     custom_mha_core_src_abs
#   ]
  
# srllm_inc_abs = [meson.current_source_dir()]
gauss_src = [  	  
  meson.current_source_dir() / 'gauss.cpp',
  meson.current_source_dir() / 'llm_util.cpp',
  meson.current_source_dir() / 'huggingface_tokenizer.cpp'
]
executable_src = [
  meson.current_source_dir() / 'main.cpp',
]

gauss_inc_abs = [meson.current_source_dir()]
gauss_inc = [include_directories('.')]

tokenizer_dependencies = []
tokenizer_path = ''

if host_machine.system() == 'windows'
  ws2_dep = cc.find_library('ws2_32', required: true)
  ntdll_dep = cc.find_library('ntdll', required: true)
  bcrypt_dep = cc.find_library('bcrypt', required: true)
  userenv_dep = cc.find_library('userenv', required: true)
  tokenizer_dependencies += ws2_dep
  tokenizer_dependencies += ntdll_dep
  tokenizer_dependencies += bcrypt_dep
  tokenizer_dependencies += userenv_dep
  tokenizer_path = meson.current_source_dir() / 'lib' / 'tokenizers_c.lib'
else
  tokenizer_path = meson.current_source_dir() / 'lib' / 'libtokenizers_c.a'
endif

if get_option('platform') != 'android'
  gauss_dependencies = [
    nntrainer_dep,
    nntrainer_ccapi_dep,
    custom_dot_wrapper_dep,
    custom_qkv_dep,
    custom_rms_norm_dep,
    custom_swiglu_dep,
    custom_fc_dep,
    custom_fc_lora_dep,
    custom_mha_core_v2_dep,    
    custom_tie_word_embedding_dep,
    openmp_dep
  ]

  if get_option('enable-fp16')
    gauss_dependencies += [custom_vocab_selection_dep]
  endif

  gauss = shared_library('gauss', 
    gauss_src,
    dependencies: [gauss_dependencies, tokenizer_dependencies],
    include_directories: gauss_inc,
    install: true,
    install_dir: application_install_dir,
    link_args: [tokenizer_path],
    cpp_args: '-DPLUGGABLE'
  )
  gauss_dep = declare_dependency(
    link_with: gauss,
    include_directories: gauss_inc
  )

  e = executable('nntrainer_attention_v2_test',
    executable_src,
    include_directories: gauss_inc,
    dependencies: [gauss_dependencies, gauss_dep],
    install: get_option('install-app'),
    install_dir: application_install_dir,
  )

else
  target_dir = meson.current_build_dir() / 'libs' / 'arm64-v8a'
  run_command('mkdir', '-p', target_dir, check: true)
  run_command('cp', '-lr', meson.current_source_dir() / 'res', meson.current_build_dir(), check: true)
  subdir('jni')
endif
