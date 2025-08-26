package com.samsung.sflare

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.animation.AnimatedVisibility
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
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
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
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.lifecycleScope
import com.samsung.sflare.ui.theme.LightGray
import com.samsung.sflare.ui.theme.NNTRBLUE
import com.samsung.sflare.ui.theme.SFlareTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch
import java.nio.charset.StandardCharsets

data class ChatMessage(val text: String, val role: String)

class MainActivity : ComponentActivity() {

    private external fun processInput(input: ByteArray)
    private external fun flareLoading()

    private val responseFlow = MutableStateFlow("")

    @Suppress("unused") // will use in jni(cpp code)
    fun onTokenReceived(tokenByte: ByteArray) {
        lifecycleScope.launch(Dispatchers.Main) {
            val token = String(tokenByte, StandardCharsets.UTF_8)
            responseFlow.emit(responseFlow.value + token)
        }
    }

    private val isInThinkMode = mutableStateOf(true)

    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        System.loadLibrary("nntrainer_engine")
        Log.i("[SFlare]", "nntrainer engine Loaded")

        flareLoading()
        Log.i("[SFlare]", "Flare Loading Done")

        enableEdgeToEdge()
        setContent {
            SFlareTheme {
                val thinkMode by remember { isInThinkMode }

                //Model Select variable
                var isModelMenuExpanded by remember { mutableStateOf(false) }
                val models = listOf("Qwen 30B-MoE", "Diffusion LLM", "OSS 20B-MoE")
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
                        }, colors = TopAppBarDefaults.topAppBarColors(
                            containerColor = Color.White, titleContentColor = Color.Black
                        ), actions = {
                            TextButton(
                                onClick = { isInThinkMode.value = !isInThinkMode.value },
                                modifier = Modifier.border(
                                    1.dp, Color.Black, RoundedCornerShape(30.dp)
                                ),
                            ) {
                                Color.LightGray
                                Box(
                                    modifier = Modifier
                                        .clip(RoundedCornerShape(10.dp))
                                        .padding(horizontal = 2.dp, vertical = 2.dp)
                                ) {
                                    Text(
                                        text = if (thinkMode) "🤔Think" else "⚡Light",
                                        color = if (thinkMode) Color.Blue else Color.Black,
                                    )
                                }
                            }
                        })
                }, content = { innerPadding ->
                    Surface(
                        modifier = Modifier
                            .fillMaxSize()
                            .padding(innerPadding),
                        color = MaterialTheme.colorScheme.background
                    ) {
                        ChatbotScreen(isInThinkMode = thinkMode)
                    }
                })
            }
        }
    }


    @Composable
    fun ChatbotScreen(isInThinkMode: Boolean) {
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
                            AssistantMessage(message.text, isInThinkMode)
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

                                val assistantPlaceholderText = if (isInThinkMode) "<think>" else ""
                                val assistantPlaceholder =
                                    ChatMessage(assistantPlaceholderText, "assistant")

                                conversationHistory =
                                    conversationHistory + userMessage + assistantPlaceholder

                                val fullPrompt = if (isInThinkMode) {
                                    "<|im_start|>user\n${inputText}<|im_end|>\n<|im_start|>assistant\n"
                                } else {
                                    "<|im_start|>user\n${inputText}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>"
                                }

                                responseFlow.value = ""
                                lifecycleScope.launch(Dispatchers.IO) {
                                    processInput(fullPrompt.toByteArray(StandardCharsets.UTF_8))
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
    fun AssistantMessage(text: String, isInThinkMode: Boolean) {
        var isThinkProcessExpanded by remember { mutableStateOf(false) }
        var isPerformanceExpanded by remember { mutableStateOf(false) }

        // Define tags and parse the incoming text
        val thinkStartTag = "<think>"
        val thinkEndTag = "</think>"
        val endTag = "<|im_end|>"
        val perfTag = "[nntrainer_perf_result]"
        val isThinking = isInThinkMode && thinkEndTag !in text


        val thinkingProcess =
            text.substringBefore(thinkEndTag, missingDelimiterValue = if (isThinking) text else "")
                .substringAfter(thinkStartTag, missingDelimiterValue = "").trim()

        val textAfterThink = text.substringAfter(thinkEndTag, "")
        val finalAnswer =
            textAfterThink.substringBefore(endTag, missingDelimiterValue = textAfterThink).trim()

        val noThinkAnswer = text.substringBefore(endTag, text).trim()

        val performanceResult = if (perfTag in text) {
            text.substringAfter(perfTag, "").trim()
        } else {
            ""
        }

        val borderColor = if (isThinking) Color.Black else Color.Blue
        val backgroundColor = Color.White
        val textColor = Color.Black
        val previewColor = Color.Black
        val CircleColor = Color.Blue

        var animatedPreviewText by remember { mutableStateOf("Thinking...") }

        LaunchedEffect(thinkingProcess, isThinking) {
            val lastLine = thinkingProcess.lines().lastOrNull { it.isNotBlank() }
            if (isThinking && !lastLine.isNullOrBlank()) {
                val words = lastLine.split(" ")
                val visibleWords = mutableListOf<String>()
                words.forEach { word ->
                    visibleWords.add(word)
                    var currentText = visibleWords.joinToString(" ")
                    val maxChars = 50
                    while (currentText.length > maxChars && visibleWords.size > 1) {
                        visibleWords.removeAt(0)
                        currentText = visibleWords.joinToString(" ")
                    }
                    animatedPreviewText = currentText

                }
            } else if (!isThinking && thinkingProcess.isNotBlank()) {
                animatedPreviewText = "Show Thought Process"
            } else {
                animatedPreviewText = "Thinking..."
            }
        }


        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(vertical = 8.dp),
            horizontalAlignment = Alignment.Start
        ) {
//            Row(
//                verticalAlignment = Alignment.CenterVertically,
//                modifier = Modifier.padding(bottom = 4.dp)
//            ) {
//                Image(
//                    painter = painterResource(id = R.drawable.flare_logo_v2),
//                    contentDescription = "SFlare Logo",
//                    modifier = Modifier.size(20.dp)
//                )
//                Spacer(modifier = Modifier.width(4.dp))

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
                    // --- Thinking Process UI ---
                    if (isInThinkMode && (isThinking || thinkingProcess.isNotBlank())) {
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(8.dp))
                                .clickable { isThinkProcessExpanded = !isThinkProcessExpanded }
                                .padding(vertical = 4.dp),
                            verticalAlignment = Alignment.CenterVertically) {
                            if (isThinking) {
                                CircularProgressIndicator(
                                    modifier = Modifier.size(20.dp),
                                    strokeWidth = 2.dp,
                                    color = CircleColor.copy(alpha = 0.7f),
                                    strokeCap = StrokeCap.Round
                                )
                                Text(
                                    text = " Thinking",
                                    color = previewColor.copy(alpha = 0.8f),
                                    fontSize = 14.sp
                                )
                            } else {
                                Icon(
                                    imageVector = Icons.Default.CheckCircle,
                                    contentDescription = "Thought Process",
                                    modifier = Modifier.size(20.dp),
                                    tint = CircleColor.copy(alpha = 0.7f)
                                )
                            }
                            Spacer(
                                modifier = Modifier
                                    .padding(horizontal = 8.dp)
                                    .height(20.dp)
                                    .width(1.dp)
                                    .background(previewColor.copy(alpha = 0.2f))
                            )

                            Text(
                                text = animatedPreviewText,
                                color = previewColor.copy(alpha = 0.6f),
                                modifier = Modifier.weight(1f),
                                maxLines = 1,
                                fontSize = 14.sp
                            )
                            Icon(
                                imageVector = if (isThinkProcessExpanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                                contentDescription = "Expand",
                                tint = previewColor.copy(alpha = 0.7f)
                            )
                        }

                        // Expandable content for the thinking process
                        AnimatedVisibility(visible = isThinkProcessExpanded) {
                            Box(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .padding(top = 4.dp, bottom = 8.dp)
                                    .clip(RoundedCornerShape(8.dp))
                                    .background(Color.Black.copy(alpha = 0.2f))
                                    .padding(8.dp)
                            ) {
                                Text(
                                    text = thinkingProcess,
                                    color = textColor.copy(alpha = 0.7f),
                                    fontSize = 14.sp
                                )
                            }
                        }
                    }

                    // Line btw Thinking & Answer
                    if (finalAnswer.isNotBlank() && thinkingProcess.isNotBlank()) {
                        Spacer(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 8.dp)
                                .height(1.dp)
                                .background(previewColor.copy(alpha = 0.2f))
                        )
                    }


                    // --- Final Answer UI ---
                    if (finalAnswer.isNotBlank()) {
                        Text(text = finalAnswer, color = textColor)
                    } else if (!isInThinkMode) {
                        // Show non-think-mode streaming text directly
                        Text(text = noThinkAnswer, color = textColor)
                    }

                    if (performanceResult.isNotBlank()) {
                        Spacer(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(vertical = 8.dp)
                                .height(1.dp)
                                .background(previewColor.copy(alpha = 0.2f))
                        )
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(8.dp))
                                .clickable { isPerformanceExpanded = !isPerformanceExpanded }
                                .padding(vertical = 4.dp),
                            verticalAlignment = Alignment.CenterVertically) {
                            Icon(
                                imageVector = Icons.Default.KeyboardArrowDown,
                                contentDescription = "Performance Result",
                                modifier = Modifier.size(20.dp),
                                tint = CircleColor.copy(alpha = 0.7f)
                            )
                            Spacer(Modifier.width(8.dp))
                            Text(
                                text = "Show Performance",
                                color = previewColor.copy(alpha = 0.8f),
                                modifier = Modifier.weight(1f),
                                fontSize = 14.sp
                            )
                            Icon(
                                imageVector = if (isThinkProcessExpanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                                contentDescription = "Expand",
                                tint = previewColor.copy(alpha = 0.7f)
                            )
                        }
                    }

                    AnimatedVisibility(visible = isPerformanceExpanded) {
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .padding(top = 4.dp)
                                .clip(RoundedCornerShape(8.dp))
                                .background(Color.Black.copy(alpha = 0.2f))
                                .padding(8.dp)

                        ) {
                            Text(
                                text = performanceResult,
                                color = textColor.copy(alpha = 0.7f),
                                fontSize = 12.sp
                            )
                        }
                    }
                }
            }
        }
    }

    @Preview(showBackground = true)
    @Composable
    fun DefaultPreview() {
        MaterialTheme {
            ChatbotScreen(isInThinkMode = true)
        }
    }
}
