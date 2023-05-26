FROM public.ecr.aws/lambda/python:3.9

RUN yum install expat curl glibc -y

COPY app.py ${LAMBDA_TASK_ROOT}
COPY nsfw_model.py ${LAMBDA_TASK_ROOT}
COPY open_nsfw_weights.h5 /var/task/open_nsfw_weights.h5
COPY open_nsfw_weights_v_1.0.0.h5 /var/task/open_nsfw_weights_v_1.0.0.h5

COPY requirements.txt .

RUN pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

CMD [ "app.lambda_handler" ]