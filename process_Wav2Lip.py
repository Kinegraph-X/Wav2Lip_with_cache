import prepare_video
import face_detect
import build_mels
import image_embeddings_preprocess
import final_processing
import time

warm_start = {

}

def process_warmed_up():
    # """
    start_time = time.perf_counter()

    # mel spectrogram may fail is the file is well formatted but too short
    try:
      _, mel_chunks = build_mels.start(warm_start["frames"])
    except Exception as e:
      return e.args
    
    final_processing.start(warm_start["frames"], mel_chunks, warm_start["face_detect_results"])
    end_time = time.perf_counter()
    print(f'Total script took {end_time - start_time}')
    return "Processing succeeded"
    # """

def process_cold_start():
    # """
    start_time = time.perf_counter()
    frames = prepare_video.start()
    warm_start["frames"] = frames
    face_detect_results = face_detect.start(frames)
    warm_start["face_detect_results"] = face_detect_results
    image_embeddings_preprocess.start(frames)

    # mel spectrogram may fail is the file is well formatted but too short
    try:
      _, mel_chunks = build_mels.start(warm_start["frames"])
    except Exception as e:
      return e.args
    
    final_processing.start(frames, mel_chunks, face_detect_results)
    end_time = time.perf_counter()
    print(f'Total script took {end_time - start_time}')
    return "Processing succeeded"
    # """

def process():
    if (len(warm_start) > 0):
        return process_warmed_up()
    else:
        return process_cold_start()