"""
Coding Nova X — Android Deployment Guide
==========================================
Prepares and documents how to deploy Coding Nova X
on Android devices using:
1. ONNX Runtime Mobile
2. TensorFlow Lite via ONNX conversion  
3. llama.cpp JNI bridge (recommended)
4. Transformers.js in WebView

Recommended path: llama.cpp → GGUF → Android JNI
This gives best performance with smallest APK size.
"""

import os
import sys
import json
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def export_to_onnx(model, tokenizer, output_path: str = "deployment/android/nova_model.onnx"):
    """
    Export model to ONNX format for Android deployment.
    
    ONNX Runtime Mobile runs on Android without Python.
    Supports INT8 quantization for small models.
    """
    import torch
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    model.eval()
    
    # Create dummy input
    dummy_ids = torch.randint(0, model.config.vocab_size, (1, 64))
    
    print(f"[ONNX] Exporting model to {output_path}...")
    print("[ONNX] This may take a few minutes...")
    
    try:
        torch.onnx.export(
            model,
            args=(dummy_ids,),
            f=output_path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq_len"},
                "logits": {0: "batch", 1: "seq_len"},
            },
            opset_version=17,
            do_constant_folding=True,
        )
        
        size_mb = os.path.getsize(output_path) / (1024*1024)
        print(f"[ONNX] ✓ Exported: {output_path} ({size_mb:.1f} MB)")
        return output_path
    
    except Exception as e:
        print(f"[ONNX] Export failed: {e}")
        print("[ONNX] Note: ONNX export requires full forward pass compatibility")
        return None


def create_android_project_guide(output_dir: str = "deployment/android"):
    """
    Generate complete Android integration guide.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    guide = {
        "title": "Coding Nova X — Android Deployment Guide",
        "recommended_approach": "llama.cpp JNI",
        "model_size_estimates": {
            "300M_FP16": "~600 MB",
            "300M_INT8": "~300 MB", 
            "300M_INT4": "~150 MB",
            "700M_INT4": "~350 MB",
        },
        "min_device_requirements": {
            "ram": "4 GB minimum, 6 GB recommended",
            "storage": "500 MB for INT4 model",
            "android_version": "Android 8.0+ (API 26)",
            "cpu": "Any ARM64 processor",
            "gpu": "Optional — Mali/Adreno for acceleration",
        },
        "deployment_steps": {
            "option_a_llamacpp": [
                "1. Quantize model to GGUF Q4_K_M format",
                "2. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp",
                "3. Build Android JNI: cd llama.cpp && ./build-android.sh",
                "4. Add llama.cpp as Android library in build.gradle",
                "5. Copy .gguf model file to assets/",
                "6. Use LlamaAndroid class for inference",
                "7. Build APK with model bundled or downloadable"
            ],
            "option_b_onnx": [
                "1. Export model to ONNX format",
                "2. Quantize ONNX model with onnxruntime tools",
                "3. Add ONNX Runtime Mobile to build.gradle",
                "4. Load model in OrtSession",
                "5. Run inference with float arrays",
                "6. Decode output IDs with tokenizer"
            ],
        },
        "gradle_dependencies": """
// Add to app/build.gradle
dependencies {
    // ONNX Runtime Mobile
    implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.16.3'
    
    // OR for llama.cpp JNI bridge
    // implementation project(':llama')
}
""",
        "kotlin_sample_code": """
// ONNX Runtime inference on Android
class CodingNovaXAndroid(private val context: Context) {
    
    private lateinit var session: OrtSession
    private lateinit var tokenizer: SimpleTokenizer
    
    fun initialize() {
        val env = OrtEnvironment.getEnvironment()
        val bytes = context.assets.open("nova_model.onnx").readBytes()
        session = env.createSession(bytes, OrtSession.SessionOptions())
        tokenizer = SimpleTokenizer.load(context)
    }
    
    fun generateCode(instruction: String, maxTokens: Int = 256): String {
        val inputIds = tokenizer.encode(instruction)
        val tensor = OnnxTensor.createTensor(env, arrayOf(inputIds))
        
        val result = session.run(mapOf("input_ids" to tensor))
        val logits = result["logits"].get().value as Array<Array<FloatArray>>
        
        val generatedIds = greedyDecode(logits, maxTokens)
        return tokenizer.decode(generatedIds)
    }
    
    private fun greedyDecode(logits: Array<Array<FloatArray>>, maxTokens: Int): IntArray {
        // Simple greedy decoding - pick highest probability token
        return logits.last().last().let { lastLogits ->
            intArrayOf(lastLogits.indices.maxByOrNull { lastLogits[it] } ?: 0)
        }
    }
}
""",
        "memory_optimization_tips": [
            "Use INT4 quantization (GGUF Q4_K_M) for smallest size",
            "Stream model weights from disk (mmap) instead of loading all to RAM",
            "Use smaller context window (512-1024) for mobile",
            "Limit batch size to 1 on mobile",
            "Use memory mapping (mmap) for model files",
            "Implement response streaming so users see output immediately",
        ],
        "performance_benchmarks": {
            "note": "Approximate, varies by device",
            "Pixel 8 (INT4)": "5-15 tokens/second",
            "Samsung S24 (INT4)": "8-20 tokens/second",
            "Mid-range Android (INT8)": "2-8 tokens/second",
        }
    }
    
    guide_path = os.path.join(output_dir, "android_deployment_guide.json")
    with open(guide_path, 'w') as f:
        json.dump(guide, f, indent=2)
    
    # Also create Kotlin sample file
    kotlin_path = os.path.join(output_dir, "CodingNovaXAndroid.kt")
    with open(kotlin_path, 'w') as f:
        f.write("""/**
 * Coding Nova X — Android Integration
 * Kotlin implementation using ONNX Runtime Mobile
 */
package com.codingnovax.android

import android.content.Context
import ai.onnxruntime.*

class CodingNovaXAndroid(private val context: Context) {
    
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var session: OrtSession
    private val tokenizer = SimpleTokenizer()
    
    fun initialize(modelAssetName: String = "nova_model.onnx") {
        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            addConfigEntry("session.use_ort_model_bytes_directly", "1")
        }
        val modelBytes = context.assets.open(modelAssetName).readBytes()
        session = ortEnv.createSession(modelBytes, sessionOptions)
        println("CodingNovaX initialized!")
    }
    
    fun generateCode(
        instruction: String,
        maxNewTokens: Int = 256,
        temperature: Float = 0.7f
    ): String {
        val inputIds = tokenizer.encode(instruction)
        val generatedIds = mutableListOf<Int>()
        
        // Autoregressive generation
        var currentInput = inputIds
        repeat(maxNewTokens) {
            val tensor = OnnxTensor.createTensor(
                ortEnv,
                arrayOf(currentInput.map { it.toLong() }.toLongArray())
            )
            
            val results = session.run(mapOf("input_ids" to tensor))
            val logits = results["logits"].get().value as Array<*>
            
            // Simple greedy decode (replace with sampling for better quality)
            val nextToken = getNextToken(logits, temperature)
            generatedIds.add(nextToken)
            currentInput = (currentInput + nextToken).toIntArray()
            
            // Stop at EOS token
            if (nextToken == tokenizer.eosId) return@repeat
        }
        
        return tokenizer.decode(generatedIds.toIntArray())
    }
    
    private fun getNextToken(logits: Array<*>, temperature: Float): Int {
        // Greedy decode for simplicity
        // TODO: Add temperature + top-p sampling
        val lastLogits = ((logits.last() as Array<*>).last() as FloatArray)
        return lastLogits.indices.maxByOrNull { lastLogits[it] } ?: 0
    }
    
    fun close() {
        session.close()
        ortEnv.close()
    }
}

/**
 * Minimal tokenizer for Android
 * In production, load vocab from assets/tokenizer.json
 */
class SimpleTokenizer {
    val eosId = 2
    val bosId = 1
    
    fun encode(text: String): IntArray {
        // Simplified - real implementation loads SentencePiece vocab
        return intArrayOf(bosId) + text.map { it.code % 32000 }.toIntArray()
    }
    
    fun decode(ids: IntArray): String {
        // Simplified - real implementation uses SentencePiece
        return ids.filter { it > 3 }.map { it.toChar() }.joinToString("")
    }
}
""")
    
    print(f"[Android] Deployment guide saved: {guide_path}")
    print(f"[Android] Kotlin sample saved: {kotlin_path}")
    return guide_path


if __name__ == "__main__":
    create_android_project_guide()
    print("\n[Android] Deployment files created!")
    print("[Android] See deployment/android/ for integration guide")
