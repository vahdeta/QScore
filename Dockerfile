FROM ubuntu:22.04

RUN apt-get update && \
    apt-get install -y wget dcm2niix python3 python3-pip

RUN apt-get install -y libgl1-mesa-dev && \
    wget https://fsl.fmrib.ox.ac.uk/fsldownloads/fslinstaller.py && \
    python3 fslinstaller.py -d /opt/fsl

RUN pip install nipype nibabel scipy numpy watchdog

# FSL env variables
ENV FSLDIR=/opt/fsl
ENV FSL_DIR=/opt/fsl
ENV FSLOUTPUTTYPE=NIFTI_GZ
ENV FSLMULTIFILEQUIT=TRUE
ENV FSLTCLSH=${FSLDIR}/bin/tclsh
ENV FSLWISH=${FSLDIR}/bin/wish
ENV FSLGECUDAQ=cuda.q
ENV FSL_LOAD_NIFTI_EXTENSIONS=0
ENV FSL_SKIP_GLOBAL=0

WORKDIR /app

COPY . /app/

ENV QSCORE_PATH /app/q_score
ENV DATA_DIR /incoming
ENV JSON_DIR /incoming

CMD ["python3","-u", "main.py"]