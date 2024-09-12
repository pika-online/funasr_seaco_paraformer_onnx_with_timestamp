# funasr_seaco_paraformer_onnx_with_timestamp
修复funasr中seaco-paraformer导出onnx后没有时间戳的bug，本仓库中funasr，funasr_onnx已经做好上述修复，直接替换使用。
onnx模型已上传modelscope: https://modelscope.cn/models/QuadraV/funasr_seaco_paraformer_onnx_with_timestamp
关于断句性能评测，请参考个人asr网站：www.funsound.cn，www.funsound.cn/whisper

### 1.问题定位
seaco-paraformer 转onnx后没有时间戳输出主要由于在定义onnx-graph时：funasr/models/seaco_paraformer/export_meta.py 没有加入时间戳预测功能
```python
  def export_backbone_forward()
  def export_backbone_output_names()
  def export_backbone_dynamic_axes()
```
对此我们参照支持时间戳功能的onnx-graph的代码：funasr/models/bicif_paraformer/export_meta.py 进行修复
```python
  # 修复1
  def export_backbone_forward()：
    ...
    # get predicted timestamps
    us_alphas, us_cif_peak = self.predictor.get_upsample_timestmap(enc, mask, pre_token_length)
    return decoder_out, pre_token_length, us_alphas, us_cif_peak

  # 修复2
  def export_backbone_output_names(self):
    return ["logits", "token_num", "us_alphas", "us_cif_peak"]

  # 修复3：
  def export_backbone_dynamic_axes(self):
    return {
        "speech": {0: "batch_size", 1: "feats_length"},
        "speech_lengths": {
            0: "batch_size",
        },
        "bias_embed": {0: "batch_size", 1: "num_hotwords"},
        "logits": {0: "batch_size", 1: "logits_length"},
        "pre_acoustic_embeds": {1: "feats_length1"},
        "us_alphas": {0: "batch_size", 1: "alphas_length"},
        "us_cif_peak": {0: "batch_size", 1: "alphas_length"},
    }
```

这样导出onnx后的就可以支持时间戳啦，当然推理代码也要调整：funasr_onnx/paraformer_bin.py 中的 ContextualParaformer.__call()__ 
```python
  try:
      outputs = self.bb_infer(feats, feats_len, bias_embed)
      am_scores, valid_token_lens = outputs[0], outputs[1]
  
      if len(outputs) == 4:
          # for BiCifParaformer Inference
          us_alphas, us_peaks = outputs[2], outputs[3]
      else:
          us_alphas, us_peaks = None, None
  
  except ONNXRuntimeError:
      # logging.warning(traceback.format_exc())
      logging.warning("input wav is silence or noise")
      preds = [""]
  else:
      preds = self.decode(am_scores, valid_token_lens)
      if us_peaks is None:
          for pred in preds:
              if self.language == "en-bpe":
                  pred = sentence_postprocess_sentencepiece(pred)
              else:
                  pred = sentence_postprocess(pred)
              asr_res.append({"preds": pred})
      else:
          for pred, us_peaks_ in zip(preds, us_peaks):
              raw_tokens = pred
              timestamp, timestamp_raw = time_stamp_lfr6_onnx(
                  us_peaks_, copy.copy(raw_tokens)
              )
              text_proc, timestamp_proc, _ = sentence_postprocess(
                  raw_tokens, timestamp_raw
              )
              # logging.warning(timestamp)
              if len(self.plot_timestamp_to):
                  self.plot_wave_timestamp(
                      waveform_list[0], timestamp, self.plot_timestamp_to
                  )
              asr_res.append(
                  {
                      "preds": text_proc,
                      "timestamp": timestamp_proc,
                      "raw_tokens": raw_tokens,
                  }
              )
```

### 2.测试

```python


# 1. 导出 seaco-paraformer onnx
from funasr import AutoModel

model = AutoModel(model='iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch')
res = model.export(quantize=False)


# 2. 查看graph
import onnx

model = onnx.load("/root/.cache/modelscope/hub/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/model.onnx")
print("Model Inputs: ", [i.name for i in model.graph.input])
print("Model Outputs: ", [i.name for i in model.graph.output])


# 3. onnxruntime 推理
from funasr_onnx import SeacoParaformer

model_dir = "/root/.cache/modelscope/hub/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
model = SeacoParaformer(model_dir, batch_size=1,quantize=False)

wav_path = "/root/.cache/modelscope/hub/iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav"
hotwords = ""

result = model(wav_path, hotwords)
print(result)

# 结果如下：
# [{'preds': '欢 迎 大 家 来 到 么 哒 社 区 进 行 体 验', 
# 'timestamp': [[990, 1290], [1290, 1610], [1610, 1830], [1830, 2010], [2010, 2170], [2170, 2430], [2430, 2570], [2570, 2850], [2850, 3050], [3050, 3390], [3390, 3570], [3570, 3910], [3910, 4110], [4110, 4345]], 'raw_tokens': ['欢', '迎', '大', '家', '来', '到', '么', '哒', '社', '区', '进', '行', '体', '验']}]
```
