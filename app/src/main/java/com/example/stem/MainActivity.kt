package com.example.stem

import androidx.compose.ui.viewinterop.AndroidView
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.border
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.media3.common.MediaItem
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                Column(
                    modifier = Modifier.fillMaxSize(),
                    verticalArrangement = Arrangement.SpaceBetween
                ) {
                    // Label for the first video
                    Text(text = "IP Address: 192.168.4.1", modifier = Modifier.padding(8.dp))
                    VideoPlayer("http://192.168.4.1")

                    Spacer(modifier = Modifier.height(16.dp))

                    // Label for the second video
                    Text(text = "IP Address: 192.168.4.2", modifier = Modifier.padding(8.dp))
                    VideoPlayer("http://192.168.4.2")
                }
            }
        }
    }
}
@Composable
fun VideoPlayer(url: String) {
    val context = LocalContext.current
    val player = remember {
        ExoPlayer.Builder(context).build().apply {
            setMediaItem(MediaItem.fromUri(Uri.parse(url)))
            prepare()
        }
    }

    AndroidView(
        modifier = Modifier
            .border(BorderStroke(2.dp, Color.Red))
            .padding(4.dp)
            .fillMaxWidth()
            .fillMaxHeight(0.5f),
        factory = { context ->
            PlayerView(context).apply {
                this.player = player
            }
        },
        update = { view ->
            view.player = player
        }
    )
}
