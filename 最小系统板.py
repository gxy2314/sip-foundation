import os
import sys
import time
import numpy as np
import torch
import librosa
import torchaudio
import soundfile as sf
import soundfile
sys.path.insert(0, './third_party/Matcha-TTS') 

from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.common import set_all_random_seed

# 模型加载
try:
    cosyvoice = CosyVoice('pretrained_models/CosyVoice2-0.5B')
except Exception:
    try:
        cosyvoice = CosyVoice2('pretrained_models/CosyVoice2-0.5B')
    except Exception:
        raise TypeError('no valid model_type!')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
# 预设采样率下限
prompt_sr = 16000
max_val = 0.8
default_data = np.zeros(cosyvoice.sample_rate)

def refresh_sft_spk():
    """刷新音色选择列表 """
    # 获取自定义音色
    files = [(entry.name, entry.stat().st_mtime) for entry in os.scandir(f"{ROOT_DIR}/voices")]
    # files.sort(key=lambda x: x[1], reverse=True) # 按时间排序
    # 添加预训练音色
    choices = [f[0].replace(".pt", "") for f in files] + cosyvoice.list_available_spks()

    if not choices:
        choices = ['']

    return {"choices": choices, "__type__": "update"}
def load_voice_data(voice_path):
    """加载音色数据"""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        voice_data = torch.load(voice_path, map_location=device) if os.path.exists(voice_path) else None
        return voice_data.get('audio_ref') if voice_data else None
    except Exception as e:
        logging.error(f"加载音色文件失败: {e}")
        return None

def validate_input(mode, tts_text, sft_dropdown, prompt_text, prompt_wav, instruct_text):
    """验证输入参数的合法性
    
    Args:
        mode: 推理模式
        tts_text: 合成文本
        sft_dropdown: 预训练音色
        prompt_text: prompt文本
        prompt_wav: prompt音频
        instruct_text: instruct文本
    
    Returns:
        bool: 验证是否通过
        str: 错误信息
    """
    if mode in ['自然语言控制']:
        if not cosyvoice.is_05b and cosyvoice.instruct is False:
            return False, f'您正在使用自然语言控制模式, 模型不支持此模式'
        if not instruct_text:
            return False, '您正在使用自然语言控制模式, 请输入instruct文本'
            
    elif mode in ['跨语种复刻']:
        if not cosyvoice.is_05b and cosyvoice.instruct is True:
            return False, f'您正在使用跨语种复刻模式, 模型不支持此模式'
        if not prompt_wav:
            return False, '您正在使用跨语种复刻模式, 请提供prompt音频'
            
    elif mode in ['3s极速复刻', '跨语种复刻']:
        if not prompt_wav:
            return False, 'prompt音频为空，您是否忘记输入prompt音频？'
        if torchaudio.info(prompt_wav).sample_rate < prompt_sr:
            return False, f'prompt音频采样率{torchaudio.info(prompt_wav).sample_rate}低于{prompt_sr}'
            
    elif mode in ['预训练音色']:
        if not sft_dropdown:
            return False, '没有可用的预训练音色！'
            
    if mode in ['3s极速复刻'] and not prompt_text:
        return False, 'prompt文本为空，您是否忘记输入prompt文本？'

    return True, ''
def postprocess(speech, top_db = 60, hop_length = 220, win_length = 440):
    """音频后处理方法"""
    # 修剪静音部分
    speech, _ = librosa.effects.trim(
        speech, 
        top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )

    # 音量归一化
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val

    # 添加尾部静音
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech

def process_audio(speech_generator, stream):
    """处理音频生成
    
    Args:
        speech_generator: 音频生成器
        stream: 是否流式处理
    
    Returns:
        tuple: (音频数据列表, 总时长)
    """
    tts_speeches = []
    total_duration = 0
    for i in speech_generator:
        tts_speeches.append(i['tts_speech'])
        total_duration += i['tts_speech'].shape[1] / cosyvoice.sample_rate
        if stream:
            yield (cosyvoice.sample_rate, i['tts_speech'].numpy().flatten()), None
            
    if not stream:
        audio_data = torch.concat(tts_speeches, dim=1)
        yield None, (cosyvoice.sample_rate, audio_data.numpy().flatten())
    
    yield total_duration

def generate_audio(tts_text, mode_checkbox_group, sft_dropdown, prompt_text, prompt_wav_upload, prompt_wav_record, instruct_text,
                   seed, stream, speed):
    """生成音频的主函数
    Args:
        tts_text: 合成文本,也就是最后读的内容
        mode_checkbox_group: 生成音频的模式
        sft_dropdown: 预训练音色
        prompt_text: prompt文本
        prompt_wav_upload: 上传的prompt音频
        prompt_wav_record: 录制的prompt音频
        instruct_text: instruct文本
        seed: 随机种子
        stream: 是否流式推理
        speed: 语速
    Yields:
        tuple: 音频数据
    """
    start_time = time.time()
    logging.info(f"开始生成音频 - 模式: {mode_checkbox_group}, 文本长度: {len(tts_text)}")
    # 处理prompt音频输入
    prompt_wav = prompt_wav_upload if prompt_wav_upload is not None else prompt_wav_record
    print(f"输入音频:{prompt_wav}")
    # 验证输入
    is_valid, error_msg = validate_input(mode_checkbox_group, tts_text, sft_dropdown, 
                                       prompt_text, prompt_wav, instruct_text)
    if not is_valid:
        print(error_msg)
        # gr.Warning(error_msg)
        # yield (cosyvoice.sample_rate, default_data), None
        return

    # 设置随机种子
    set_all_random_seed(seed)

    # 根据不同模式处理
    if mode_checkbox_group == '预训练音色':
        # logging.info('get sft inference request')
        generator = cosyvoice.inference_sft(tts_text, sft_dropdown, stream=False, speed=speed)
        
    elif mode_checkbox_group in ['3s极速复刻', '跨语种复刻']:
        # logging.info(f'get {mode_checkbox_group} inference request')
        prompt_speech_16k = postprocess(load_wav(prompt_wav, prompt_sr))
        inference_func = (cosyvoice.inference_zero_shot if mode_checkbox_group == '3s极速复刻' 
                         else cosyvoice.inference_cross_lingual)
        generator = inference_func(tts_text, prompt_text, prompt_speech_16k, stream=False, speed=speed)
        
    else:  # 自然语言控制模式
        # logging.info('get instruct inference request')
        voice_path = f"{ROOT_DIR}/voices/{sft_dropdown}.pt"
        prompt_speech_16k = load_voice_data(voice_path)
        
        if prompt_speech_16k is None:
            # gr.Warning('预训练音色文件中缺少prompt_speech数据！')
            print('预训练音色文件中缺少prompt_speech数据！')
            # yield (cosyvoice.sample_rate, default_data), None
            return
            
        generator = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_speech_16k, 
                                                stream=stream, speed=speed)

    # 处理音频生成并获取总时长
    audio_generator = process_audio(generator, stream)
    total_duration = 0
    
    # 收集所有音频输出
    for output in audio_generator:
        if isinstance(output, (float, int)):  # 如果是总时长
            total_duration = output
        else:  # 如果是音频数据
            # yield output
            return output

    processing_time = time.time() - start_time
    rtf = processing_time / total_duration if total_duration > 0 else 0
    logging.info(f"音频生成完成 耗时: {processing_time:.2f}秒, rtf: {rtf:.2f}")

if __name__ == '__main__':
    tts_text = "make xiaoniuma great again"
    mode_checkbox_group = "跨语种复刻"# 可选 自然语言控制 预训练音色 跨语种复刻 
    sft_dropdown = refresh_sft_spk()['choices'][5]
    prompt_text = ""
    prompt_wav_upload = None
    prompt_wav_record = 'audios/麦克阿瑟.mp3'
    instruct_text = "用西海岸方言说这句话"# 也就是对它的要求，自然语言控制模式下必须
    seed = 42
    stream = False
    speed = 1.0
    generator = generate_audio(tts_text, mode_checkbox_group, sft_dropdown,
                           prompt_text, prompt_wav_upload, prompt_wav_record,
                           instruct_text, seed, stream, speed)
    print("返回值类型：", type(generator), "内容：", generator)
    # 获取音频输出
    for result in generator:
        if isinstance(result, tuple) and isinstance(result[0], int):
            sample_rate, waveform = result
            print(f"采样率: {sample_rate}, 音频形状: {waveform.shape}")
            # 播放音频
            # 保存音频
            sf.write("output/output跨语种.wav", waveform, sample_rate)