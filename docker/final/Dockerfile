FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV MVAA_INPUT_DIR=/input
ENV MVAA_OUTPUT_DIR=/output
ENV MVAA_WORK_DIR=/work
ENV MVAA_NO_PRETRAINED_INIT=1
ENV TORCH_CUDA_ARCH_LIST=7.0

WORKDIR /workspace

ARG PIP_INDEX_URL=https://pypi.org/simple

COPY docker/final/requirements.txt /workspace/docker/final/requirements.txt
RUN python -m pip install --upgrade pip \
        --index-url "${PIP_INDEX_URL}" \
    && python -m pip install -r /workspace/docker/final/requirements.txt \
        --index-url "${PIP_INDEX_URL}"

COPY task3/ /workspace/task3/
COPY scripts/final_infer_v39_safe.py /workspace/scripts/final_infer_v39_safe.py
COPY docker/final/run.sh /workspace/run.sh

COPY outputs/nnunet_task1_pseudo_top100/nnUNet_results/ /workspace/weights/nnunet_task1_v25/nnUNet_results/
COPY outputs/nnunet_task2_fold0/nnUNet_results/ /workspace/weights/nnunet_task2_v19/nnUNet_results/
COPY outputs/exp/task3_unetpp_res34_imagenet_e100_bs2_bestcfg/checkpoints/best.pt /workspace/weights/task3_v39/resnet34_s42_best.pt
COPY outputs/opt/task3/t3_v36_unetpp_res34_img448x800_s49/checkpoints/best.pt /workspace/weights/task3_v39/resnet34_s49_best.pt
COPY outputs/opt/task3/t3_sup_unetpp_effb4_img512x896_allframe_s42/checkpoints/best.pt /workspace/weights/task3_v39/effb4_s42_best.pt
COPY outputs/opt/task3/t3_v32_unetpp_effb5_sup_allframe_s42/checkpoints/best.pt /workspace/weights/task3_v39/effb5_s42_best.pt
COPY outputs/opt/task3/t3_v35_unetpp_effb5_sup_allframe_s44/checkpoints/best.pt /workspace/weights/task3_v39/effb5_s44_best.pt
COPY outputs/opt/task3/t3_lemonfm_fpn_formal_tail2_s52/checkpoints/best.pt /workspace/weights/task3_v39/lemonfm_fpn_s52_best.pt
COPY outputs/opt/task3/t3_lemonfm_presence_gate_s51/checkpoints/best.pt /workspace/weights/task3_v39/lemonfm_presence_gate_s51_best.pt

RUN chmod +x /workspace/run.sh

ENTRYPOINT ["/workspace/run.sh"]
