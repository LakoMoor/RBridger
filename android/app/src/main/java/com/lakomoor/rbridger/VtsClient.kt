package com.lakomoor.rbridger

import android.os.Handler
import android.os.Looper
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID
import java.util.concurrent.TimeUnit

enum class ConnState { DISCONNECTED, CONNECTING, AUTHENTICATING, CONNECTED, ERROR }

class VtsClient(private val onState: (ConnState) -> Unit) {

    private val http = OkHttpClient.Builder()
        .pingInterval(15, TimeUnit.SECONDS)
        .connectTimeout(10, TimeUnit.SECONDS)
        .build()

    private var ws: WebSocket? = null
    private var token: String? = null
    private val mainHandler = Handler(Looper.getMainLooper())

    var state = ConnState.DISCONNECTED
        private set(v) { field = v; mainHandler.post { onState(v) } }

    fun connect(host: String, port: Int = 8001) {
        if (state != ConnState.DISCONNECTED && state != ConnState.ERROR) return
        setState(ConnState.CONNECTING)
        val req = Request.Builder().url("ws://$host:$port").build()
        ws = http.newWebSocket(req, Listener())
    }

    fun disconnect() {
        ws?.close(1000, null)
        ws = null
        token = null
        setState(ConnState.DISCONNECTED)
    }

    fun injectFace(data: FaceData) {
        if (state != ConnState.CONNECTED) return
        val params = JSONArray()
        data.blendShapes.forEach { (name, value) ->
            params.put(JSONObject().put("id", name).put("value", value.toDouble()))
        }
        mapOf("FaceAngleX" to data.pitch, "FaceAngleY" to data.yaw, "FaceAngleZ" to data.roll)
            .forEach { (id, v) -> params.put(JSONObject().put("id", id).put("value", v.toDouble())) }

        send(vtsMsg("InjectParameterDataRequest", JSONObject()
            .put("faceFound", true)
            .put("mode", "set")
            .put("parameterValues", params)))
    }

    private inner class Listener : WebSocketListener() {
        override fun onOpen(ws: WebSocket, response: Response) {
            setState(ConnState.AUTHENTICATING)
            if (token != null) authenticate(token!!) else requestToken()
        }

        override fun onMessage(ws: WebSocket, text: String) {
            val json = runCatching { JSONObject(text) }.getOrNull() ?: return
            when (json.optString("messageType")) {
                "AuthenticationTokenResponse" -> {
                    token = json.getJSONObject("data").getString("authenticationToken")
                    authenticate(token!!)
                }
                "AuthenticationResponse" -> {
                    val ok = json.getJSONObject("data").getBoolean("authenticated")
                    setState(if (ok) ConnState.CONNECTED else ConnState.ERROR)
                }
                "APIError" -> setState(ConnState.ERROR)
            }
        }

        override fun onFailure(ws: WebSocket, t: Throwable, r: Response?) {
            setState(ConnState.ERROR)
        }

        override fun onClosed(ws: WebSocket, code: Int, reason: String) {
            setState(ConnState.DISCONNECTED)
        }
    }

    private fun requestToken() = send(vtsMsg("AuthenticationTokenRequest", JSONObject()
        .put("pluginName", "RBridger Android")
        .put("pluginDeveloper", "LakoMoor")
        .put("pluginIcon", "")))

    private fun authenticate(tok: String) = send(vtsMsg("AuthenticationRequest", JSONObject()
        .put("pluginName", "RBridger Android")
        .put("pluginDeveloper", "LakoMoor")
        .put("authenticationToken", tok)))

    private fun vtsMsg(type: String, data: JSONObject) = JSONObject()
        .put("apiName", "VTubeStudioPublicAPI")
        .put("apiVersion", "1.0")
        .put("requestID", UUID.randomUUID().toString())
        .put("messageType", type)
        .put("data", data)
        .toString()

    private fun send(text: String) = ws?.send(text)

    private fun setState(s: ConnState) { state = s }
}
