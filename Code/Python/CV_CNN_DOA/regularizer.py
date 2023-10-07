# -*- coding: utf-8 -*-
__author__ = 'maoss2'
# https://github.com/dizam92/pyTorchReg/blob/53e8e562abd12478fccc7c628035ec93d477d9f0/src/digits_experience_with_reg.py#L78
import torch
import torch.nn as nn
class SparseOutputLoss(nn.Module):
    def __init__(self, base_loss, l1_weight=0.01):
        super(SparseOutputLoss, self).__init__()
        self.base_loss = base_loss  # 原始损失函数，例如 nn.MSELoss()
        self.l1_weight = l1_weight  # L1正则化权重

    def forward(self, output, target):
        # 计算原始损失
        base_loss_value = self.base_loss(output, target)
        # 计算L1范数
        l1_norm = torch.norm(output, p=1)
        # 总损失
        total_loss = base_loss_value + self.l1_weight * l1_norm
        return total_loss
class _Regularizer(object):
    """
    Parent class of Regularizers
    """

    def __init__(self, model):
        super(_Regularizer, self).__init__()
        self.model = model

    def regularized_param(self, param_weights, reg_loss_function):
        raise NotImplementedError

    def regularized_all_param(self, reg_loss_function):
        raise NotImplementedError


class L1Regularizer(_Regularizer):
    """
    L1 regularized loss
    """
    def __init__(self, model, lambda_reg=0.001):
        super(L1Regularizer, self).__init__(model=model)
        self.lambda_reg = lambda_reg

    def regularized_param(self, param_weights, reg_loss_function):
        reg_loss_function += self.lambda_reg * L1Regularizer.__add_l1(var=param_weights)
        return reg_loss_function

    def regularized_all_param(self, reg_loss_function):
        for model_param_name, model_param_value in self.model.named_parameters():
            if model_param_name.endswith('weight'):
                reg_loss_function += self.lambda_reg * L1Regularizer.__add_l1(var=model_param_value)
        return reg_loss_function

    @staticmethod
    def __add_l1(var):
        return var.abs().sum()


class L2Regularizer(_Regularizer):
    """
       L2 regularized loss
    """
    def __init__(self, model, lambda_reg=0.01):
        super(L2Regularizer, self).__init__(model=model)
        self.lambda_reg = lambda_reg

    def regularized_param(self, param_weights, reg_loss_function):
        reg_loss_function += self.lambda_reg * L2Regularizer.__add_l2(var=param_weights)
        return reg_loss_function

    def regularized_all_param(self, reg_loss_function):
        for model_param_name, model_param_value in self.model.named_parameters():
            if model_param_name.endswith('weight'):
                reg_loss_function += self.lambda_reg * L2Regularizer.__add_l2(var=model_param_value)
        return reg_loss_function

    @staticmethod
    def __add_l2(var):
        return var.pow(2).sum()


class ElasticNetRegularizer(_Regularizer):
    """
    Elastic Net Regularizer
    """
    def __init__(self, model, lambda_reg=0.01, alpha_reg=0.01):
        super(ElasticNetRegularizer, self).__init__(model=model)
        self.lambda_reg = lambda_reg
        self.alpha_reg = alpha_reg

    def regularized_param(self, param_weights, reg_loss_function):
        reg_loss_function += self.lambda_reg * \
                                     (((1 - self.alpha_reg) * ElasticNetRegularizer.__add_l2(var=param_weights)) +
                                      (self.alpha_reg * ElasticNetRegularizer.__add_l1(var=param_weights)))
        return reg_loss_function

    def regularized_all_param(self, reg_loss_function):
        for model_param_name, model_param_value in self.model.named_parameters():
            if model_param_name.endswith('weight'):
                reg_loss_function += self.lambda_reg * \
                                 (((1 - self.alpha_reg) * ElasticNetRegularizer.__add_l2(var=model_param_value)) +
                                  (self.alpha_reg * ElasticNetRegularizer.__add_l1(var=model_param_value)))
        return reg_loss_function

    @staticmethod
    def __add_l1(var):
        return var.abs().sum()

    @staticmethod
    def __add_l2(var):
        return var.pow(2).sum()


class GroupSparseLassoRegularizer(_Regularizer):
    """
    Group Sparse Lasso Regularizer
    """
    def __init__(self, model, lambda_reg=0.01):
        super(GroupSparseLassoRegularizer, self).__init__(model=model)
        self.lambda_reg = lambda_reg
        self.reg_l2_l1 = GroupLassoRegularizer(model=self.model, lambda_reg=self.lambda_reg)
        self.reg_l1 = L1Regularizer(model=self.model, lambda_reg=self.lambda_reg)

    def regularized_param(self, param_weights, reg_loss_function):
        reg_loss_function = self.lambda_reg * (
                    self.reg_l2_l1.regularized_param(param_weights=param_weights, reg_loss_function=reg_loss_function)
                    + self.reg_l1.regularized_param(param_weights=param_weights, reg_loss_function=reg_loss_function))

        return reg_loss_function

    def regularized_all_param(self, reg_loss_function):
        reg_loss_function = self.lambda_reg * (
                self.reg_l2_l1.regularized_all_param(reg_loss_function=reg_loss_function)
                + self.reg_l1.regularized_all_param(reg_loss_function=reg_loss_function))

        return reg_loss_function


class GroupLassoRegularizer(_Regularizer):
    """
    GroupLasso Regularizer:
    The first dimension represents the input layer and the second dimension represents the output layer.
    The groups are defined by the column in the matrix W
    """
    def __init__(self, model, lambda_reg=0.01):
        super(GroupLassoRegularizer, self).__init__(model=model)
        self.lambda_reg = lambda_reg

    def regularized_param(self, param_weights, reg_loss_function, group_name='input_group'):
        if group_name == 'input_group':
            reg_loss_function += self.lambda_reg * GroupLassoRegularizer.__inputs_groups_reg(
                        layer_weights=param_weights)  # apply the group norm on the input value
        elif group_name == 'hidden_group':
            reg_loss_function += self.lambda_reg * GroupLassoRegularizer.__inputs_groups_reg(
                        layer_weights=param_weights)  # apply the group norm on every hidden layer
        elif group_name == 'bias_group':
            reg_loss_function += self.lambda_reg * GroupLassoRegularizer.__bias_groups_reg(
                        bias_weights=param_weights)  # apply the group norm on the bias
        else:
            print(
                'The group {} is not supported yet. Please try one of this: [input_group, hidden_group, bias_group]'.format(
                    group_name))
        return reg_loss_function

    def regularized_all_param(self, reg_loss_function):
        for model_param_name, model_param_value in self.model.named_parameters():
            if model_param_name.endswith('weight'):
                reg_loss_function += self.lambda_reg * GroupLassoRegularizer.__inputs_groups_reg(
                    layer_weights=model_param_value)
            if model_param_name.endswith('bias'):
                    reg_loss_function += self.lambda_reg * GroupLassoRegularizer.__bias_groups_reg(
                        bias_weights=model_param_value)
        return reg_loss_function

    @staticmethod
    def __grouplasso_reg(groups, dim):
        if dim == -1:
            # We only have single group
            return groups.norm(2)
        return groups.norm(2, dim=dim).sum()

    @staticmethod
    def __inputs_groups_reg(layer_weights):
        return GroupLassoRegularizer.__grouplasso_reg(groups=layer_weights, dim=1)

    @staticmethod
    def __bias_groups_reg(bias_weights):
        return GroupLassoRegularizer.__grouplasso_reg(groups=bias_weights, dim=-1)  # ou 0 i dont know yet