package com.samsung.sflare

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.lifecycleScope
import com.samsung.sflare.ui.theme.DarkGray
import com.samsung.sflare.ui.theme.LightGray
import com.samsung.sflare.ui.theme.NNTRBLUE
import com.samsung.sflare.ui.theme.TextGray
import com.samsung.sflare.ui.theme.topbar_color
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch

data class ChatMessage(val text: String, val role: String)

class MainActivity : ComponentActivity() {

    private external fun processInput(input: String, app_path: String)

    private val responseFlow = MutableStateFlow("Output will appear Here")

    @Suppress("unused") // will use in jni(cpp code)
    fun onTokenReceived(token: String) {
        lifecycleScope.launch(Dispatchers.Main) {
            responseFlow.emit(responseFlow.value + token)
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        System.loadLibrary("nntrainer_engine")
        Log.e("[SFlare]", "nntrainer engine Loaded")

        enableEdgeToEdge()
        setContent {
            MaterialTheme {
                Scaffold(topBar = {
                    TopAppBar(
                        title = { Text("SFlare") }, colors = TopAppBarDefaults.topAppBarColors(
                            containerColor = topbar_color, titleContentColor = Color.White
                        )
                    )
                }, content = { innerPadding ->
                    Surface(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(innerPadding),
                        color = MaterialTheme.colorScheme.background
                    ) {
                        ChatbotScreen()
                    }
                })
            }
        }
    }


    @Composable
    fun ChatbotScreen() {
        var inputText by remember { mutableStateOf("Give me a short introduction to large language model.") }
        val currentResponse by responseFlow.collectAsStateWithLifecycle()

        // 1. 대화 기록을 ChatMessage 리스트로 관리
        var conversationHistory by remember { mutableStateOf<List<ChatMessage>>(listOf()) }
        val listState = rememberLazyListState()

        // JNI에서 새로운 토큰이 들어올 때마다 마지막 메시지 업데이트
        LaunchedEffect(currentResponse) {
            if (currentResponse.isNotBlank()) {
                val lastMessage = conversationHistory.lastOrNull()
                // 마지막 메시지가 AI의 답변일 경우, 내용을 덧붙여서 업데이트
                if (lastMessage != null && lastMessage.role == "assistant") {
                    conversationHistory =
                        conversationHistory.dropLast(1) + lastMessage.copy(text = currentResponse)
                }
            }
        }

        // 새 메시지가 추가되면 자동으로 스크롤
        LaunchedEffect(conversationHistory.size) {
            if (conversationHistory.isNotEmpty()) {
                listState.animateScrollToItem(conversationHistory.lastIndex)
            }
        }

        Surface(modifier = Modifier.fillMaxSize(), color = DarkGray) {
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 16.dp, vertical = 8.dp)
            ) {
                // 대화 내용 표시
                LazyColumn(
                    state = listState,
                    modifier = Modifier.weight(1f),
                    contentPadding = PaddingValues(vertical = 8.dp)
                ) {
                    items(conversationHistory) { message ->
                        if (message.role == "user") {
                            UserMessage(message.text)
                        } else {
                            AssistantMessage(message.text)
                        }
                    }
                }

                // 하단 입력창
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(top = 8.dp, bottom = 8.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    TextField(
                        value = inputText,
                        onValueChange = { inputText = it },
                        placeholder = { Text("메시지를 입력하세요", color = TextGray) },
                        modifier = Modifier
                            .weight(1f)
                            .clip(RoundedCornerShape(24.dp))
                            .background(LightGray),
                        colors = TextFieldDefaults.colors(
                            focusedTextColor = Color.White,
                            unfocusedTextColor = Color.White,
                            cursorColor = NNTRBLUE,
                            focusedIndicatorColor = Color.Transparent,
                            unfocusedIndicatorColor = Color.Transparent
                        )
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Button(
                        onClick = {
                            if (inputText.isNotBlank()) {
                                val userMessage = ChatMessage(inputText, "user")
                                val assistantPlaceholder = ChatMessage("", "assistant")

                                // 2. 사용자 질문과 AI 답변 placeholder를 대화 기록에 추가
                                conversationHistory =
                                    conversationHistory + userMessage + assistantPlaceholder

                                val fullPrompt =
                                    "<|im_start|>user\n${inputText}<|im_end|>\n<|im_start|>assistant\n"

                                responseFlow.value = "" // 답변 스트림 초기화
                                lifecycleScope.launch(Dispatchers.IO) {
                                    val appPath = filesDir.absolutePath
                                    processInput(fullPrompt, appPath)
                                }
                                inputText = "" // 입력창 비우기
                            }
                        },
                        modifier = Modifier.size(48.dp),
                        shape = CircleShape,
                        contentPadding = PaddingValues(0.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = LightGray)
                    ) {
                        Icon(
                            imageVector = Icons.Default.KeyboardArrowUp,
                            contentDescription = "Send",
                            tint = if (inputText.isNotBlank()) NNTRBLUE else TextGray
                        )
                    }
                }
            }
        }
    }

    //User Message Box
    @Composable
    fun UserMessage(text: String) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp),
            horizontalAlignment = Alignment.End // 오른쪽 정렬
        ) {
            Text(
                text = "You",
                color = Color.White,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(bottom = 4.dp)
            )
            Box(
                modifier = Modifier
                    .clip(
                        RoundedCornerShape(
                            topStart = 16.dp, topEnd = 4.dp, bottomStart = 16.dp, bottomEnd = 16.dp
                        )
                    )
                    .background(NNTRBLUE) // 사용자 메시지 배경색
                    .padding(horizontal = 12.dp, vertical = 8.dp)
            ) {
                Text(text = text, color = Color.Black)
            }
        }
    }

    // Flare Answer Box
    @Composable
    fun AssistantMessage(text: String) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp),
            horizontalAlignment = Alignment.Start
        ) {
            Text(
                text = "SFlare", // Name
                color = Color.White,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(bottom = 4.dp)
            )
            Box(
                modifier = Modifier
                    .clip(
                        RoundedCornerShape(
                            topStart = 4.dp, topEnd = 16.dp, bottomStart = 16.dp, bottomEnd = 16.dp
                        )
                    )
                    .background(LightGray) // AI Response Background Color
                    .padding(horizontal = 12.dp, vertical = 8.dp)
            ) {
                Text(text = text, color = TextGray)
            }
        }
    }

    @Preview(showBackground = true)
    @Composable
    fun DefaultPreview() {
        MaterialTheme {
            ChatbotScreen()
        }
    }

}