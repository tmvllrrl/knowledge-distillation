# Knowledge Distillation

This code is a replication of the knowledge distillation technique discussed in [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) using a ResNet-50 model, initialized with pre-trained weights on IN1k and then fine-tuned on CIFAR10, as the teacher model and various versions of a ResNet-18 model as the student.

For comparison purposes, 4 ResNet-18 models were trained: 2 using knowledge distillation and 2 through regular classification training on CIFAR10. The 2 ResNet-18 models trained using knowledge distillation either had randomly initialized weights (KD-RN18-rand) or had weights transferred from a pre-trained ResNet-18 on IN1k (KD-RN18-in1k). Similarly, the 2 models trained through regular classification training either had the ResNet-18 model's weights randomly intialized (RN18-rand) or initialized using a pre-trained ResNet-18 on IN1k (RN18-in1k). For the knowledge distillation models, only the soft loss term was used as additionally using the hard loss term resulted in degraded performance from the experiments. For the regularly trained models, the hard term was only used. All 4 models used the Adam optimizer with cross entropy loss. Average classification accuracy with standard deviation on the test set of CIFAR10 are given below for all 5 models (teacher + 4 student models).

| Model               | Accuracy (%)  |
|---------------------|---------------|
| ResNet-50 (Teacher) | 89.2          |
| RN18-rand           | 78.4 ± 0.0028 |
| KD-RN18-rand        | 80.1 ± 0.0049 |
| RN18-in1k           | 84.8 ± 0.0004 |
| KD-RN18-in1k        | 86.2 ± 0.0014 |

The averages and standard deviations were computed using 3 random runs for each model.

We can see from the results that knowledge distillation results in significantly better performance compared to regular classification training.