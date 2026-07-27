FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV MVAA_INPUT_DIR=/input
ENV MVAA_OUTPUT_DIR=/output
ENV MVAA_WORK_DIR=/work
ENV MVAA_NO_PRETRAINED_INIT=1
ENV TORCH_CUDA_ARCH_LIST=7.0

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/final/requirements.txt /workspace/docker/final/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /workspace/docker/final/requirements.txt

COPY task3/ /workspace/task3/
COPY scripts/final_infer_v35.py /workspace/scripts/final_infer_v35.py
COPY docker/final/run.sh /workspace/run.sh

COPY outputs/nnunet_task1_pseudo_top100/nnUNet_results/ /workspace/weights/nnunet_task1_v25/nnUNet_results/
COPY outputs/nnunet_task2_fold0/nnUNet_results/ /workspace/weights/nnunet_task2_v19/nnUNet_results/
COPY outputs/exp/task3_unetpp_res34_imagenet_e100_bs2_bestcfg/checkpoints/best.pt /workspace/weights/task3_v35/resnet34_s42_best.pt
COPY outputs/opt/task3/t3_v36_unetpp_res34_img448x800_s49/checkpoints/best.pt /workspace/weights/task3_v35/resnet34_s49_best.pt
COPY outputs/opt/task3/t3_sup_unetpp_effb4_img512x896_allframe_s42/checkpoints/best.pt /workspace/weights/task3_v35/effb4_s42_best.pt
COPY outputs/opt/task3/t3_v32_unetpp_effb5_sup_allframe_s42/checkpoints/best.pt /workspace/weights/task3_v35/effb5_s42_best.pt
COPY outputs/opt/task3/t3_v35_unetpp_effb5_sup_allframe_s44/checkpoints/best.pt /workspace/weights/task3_v35/effb5_s44_best.pt

RUN chmod +x /workspace/run.sh

ENTRYPOINT ["/workspace/run.sh"]
