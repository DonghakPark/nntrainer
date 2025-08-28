package com.samsung.flare_diffusion

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsFocusedAsState
import androidx.compose.foundation.layout.Arrangement
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
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
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.lifecycleScope
import com.samsung.flare_diffusion.ui.theme.Flare_DiffusionTheme
import com.samsung.flare_diffusion.ui.theme.LightGray
import com.samsung.flare_diffusion.ui.theme.NNTRBLUE
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch
import java.nio.charset.StandardCharsets

data class ChatMessage(val text: String, val role: String)

class MainActivity : ComponentActivity() {

    // Diffusion Cpp JNI Function
    private external fun DiffusionRun(input: ByteArray)

    private val responseFlow = MutableStateFlow("")

    @Suppress("unused") // will use in Qwen(cpp code)
    fun onTokenReceived(tokenByte: ByteArray) {
        lifecycleScope.launch(Dispatchers.Main) {
            val token = String(tokenByte, StandardCharsets.UTF_8)
            responseFlow.emit(responseFlow.value + token)
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        System.loadLibrary("diffusion_engine")
        Log.i("[SFlare]", "diffusion engine Loaded")

        enableEdgeToEdge()
        setContent {
            Flare_DiffusionTheme {

                //Model Select variable
                var isModelMenuExpanded by remember { mutableStateOf(false) }
                val models = listOf("Diffusion LLM")
                var selectedModel by remember { mutableStateOf(models[0]) }

                Scaffold(topBar = {
                    TopAppBar(
                        title = {
                            Box(
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                //App Title (SFlare)

                                Text(
                                    text = "SFlare",
                                    color = Color.Black,
                                    fontWeight = FontWeight.Bold,
                                    fontSize = 20.sp,
                                    modifier = Modifier.align(Alignment.CenterStart)
                                )
                                Box(modifier = Modifier.align(Alignment.Center)) {
                                    Row(
                                        modifier = Modifier.clickable {
                                            isModelMenuExpanded = true
                                        }, verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        Text(
                                            text = selectedModel,
                                            color = Color.Black,
                                            fontWeight = FontWeight.SemiBold,
                                            fontSize = 16.sp
                                        )
                                        Spacer(modifier = Modifier.width(4.dp))
                                        Icon(
                                            imageVector = if (isModelMenuExpanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                                            contentDescription = "Select Model",
                                            tint = Color.Gray,
                                            modifier = Modifier.size(20.dp)
                                        )
                                    }
                                    DropdownMenu(
                                        expanded = isModelMenuExpanded,
                                        onDismissRequest = { isModelMenuExpanded = false },
                                        modifier = Modifier.background(color = LightGray.copy(alpha = 0.5f))

                                    ) {
                                        models.forEach { model ->
                                            DropdownMenuItem(text = { Text(model) }, onClick = {
                                                selectedModel = model
                                                isModelMenuExpanded = false
                                                //@TODO model select Logic
                                            })
                                        }
                                    }
                                }

                            }
                        },
                        colors = TopAppBarDefaults.topAppBarColors(
                            containerColor = Color.White, titleContentColor = Color.Black
                        ),
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
        var inputText by remember {
            mutableStateOf(
                "Give me a short introduction to large language model."
            )
        }
        val currentResponse by responseFlow.collectAsStateWithLifecycle()
        var conversationHistory by remember { mutableStateOf<List<ChatMessage>>(listOf()) }
        val scrollState = rememberScrollState()

        // This effect updates the last message in the history with the new streaming content.
        LaunchedEffect(currentResponse) {
            if (currentResponse.isNotBlank()) {
                val lastMessage = conversationHistory.lastOrNull()
                if (lastMessage != null && lastMessage.role == "assistant") {
                    conversationHistory =
                        conversationHistory.dropLast(1) + lastMessage.copy(text = currentResponse)
                }
            }
        }

        // This effect scrolls to the bottom when the conversation updates.
        LaunchedEffect(conversationHistory.size, currentResponse) {
            scrollState.animateScrollTo(scrollState.maxValue)
        }

        val interactionSource = remember { MutableInteractionSource() }
        val isTextFieldFocused by interactionSource.collectIsFocusedAsState()
        LaunchedEffect(isTextFieldFocused) {
            delay(200)
            scrollState.animateScrollTo(scrollState.maxValue)
        }

        Surface(modifier = Modifier.fillMaxSize(), color = Color.White) {
            Box(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(horizontal = 16.dp)
            ) {
                Column(
                    modifier = Modifier
                        .fillMaxSize()
                        .padding(bottom = 80.dp)
                        .verticalScroll(scrollState),
//                        .then(pointerInputModifier),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Spacer(modifier = Modifier.weight(1f))
                    conversationHistory.forEach { message ->
                        if (message.role == "user") {
                            UserMessage(message.text)
                        } else {
                            // Pass the full text to AssistantMessage, it will handle the parsing.
                            AssistantMessage(message.text)
                        }
                    }
                }

                Row(
                    modifier = Modifier
                        .align(Alignment.BottomCenter)
                        .fillMaxWidth()
                        .padding(top = 8.dp, bottom = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    TextField(
                        value = inputText,
                        onValueChange = { inputText = it },
                        placeholder = { Text("Ask Anything", color = Color.Black) },
                        modifier = Modifier
                            .weight(1f)
                            .clip(RoundedCornerShape(24.dp))
                            .background(Color.White)
                            .border(1.dp, Color.Gray.copy(alpha = 0.4f), RoundedCornerShape(24.dp)),
                        colors = TextFieldDefaults.colors(
                            focusedTextColor = Color.Black,
                            unfocusedTextColor = Color.Black,
                            cursorColor = Color.Black,
                            focusedIndicatorColor = Color.Transparent,
                            unfocusedIndicatorColor = Color.Transparent,

                            unfocusedContainerColor = Color.Transparent,
                            focusedContainerColor = Color.Transparent

                        )
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Button(
                        onClick = {
                            if (inputText.isNotBlank()) {
                                val userMessage = ChatMessage(inputText, "user")

                                val assistantPlaceholder = ChatMessage("", "assistant")

                                conversationHistory =
                                    conversationHistory + userMessage + assistantPlaceholder

                                val fullPrompt =
                                    "<|im_start|>user\n${inputText}<|im_end|>\n<|im_start|>assistant\n"


                                responseFlow.value = ""
                                lifecycleScope.launch(Dispatchers.IO) {
                                    DiffusionRun(fullPrompt.toByteArray(StandardCharsets.UTF_8))
                                }
                                inputText = ""
                            }
                        },
                        modifier = Modifier
                            .size(48.dp)
                            .border(1.dp, Color.Gray.copy(alpha = 0.4f), CircleShape),
                        shape = CircleShape,
                        contentPadding = PaddingValues(0.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = Color.White)
                    ) {
                        Icon(
                            imageVector = Icons.Default.KeyboardArrowUp,
                            contentDescription = "Send",
                            tint = if (inputText.isNotBlank()) Color.Blue else Color.Black
                        )
                    }
                }
//                }
            }
        }
    }

    @Composable
    fun UserMessage(text: String) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp),
            horizontalAlignment = Alignment.End
        ) {
            Text(
                text = "You",
                color = Color.Black,
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
                    .background(NNTRBLUE)
                    .padding(horizontal = 12.dp, vertical = 8.dp)
            ) {
                Text(text = text, color = Color.Black)
            }
        }
    }

    // --- ENTIRELY REWRITTEN AssistantMessage ---
    @Composable
    fun AssistantMessage(text: String) {

        // Define tags and parse the incoming text
        val noThinkAnswer = text.trim()

        val borderColor = Color.Blue
        val backgroundColor = Color.White
        val textColor = Color.Black
        Color.Black
        Color.Blue

        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp),
            horizontalAlignment = Alignment.Start
        ) {
            Text(
                text = "SFlare",
                color = Color.Black,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.padding(bottom = 4.dp)
            )
//            }
            Box(
                modifier = Modifier
                    .clip(
                        RoundedCornerShape(
                            topStart = 4.dp, topEnd = 16.dp, bottomStart = 16.dp, bottomEnd = 16.dp
                        )
                    )
                    .border(
                        width = 1.5.dp, color = borderColor, shape = RoundedCornerShape(
                            topStart = 4.dp, topEnd = 16.dp, bottomStart = 16.dp, bottomEnd = 16.dp
                        )

                    )
                    .background(backgroundColor)
                    .padding(horizontal = 12.dp, vertical = 8.dp)
            ) {
                Column(modifier = Modifier.fillMaxWidth()) {

                    // --- Final Answer UI ---
                    Text(text = noThinkAnswer, color = textColor)
                }
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
