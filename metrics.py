# Copyright (c) Facebook, Inc. and its affiliates.
"""
The metrics module contains implementations of various metrics used commonly to
understand how well our models are performing. For e.g. accuracy, vqa_accuracy,
r@1 etc.

For implementing your own metric, you need to follow these steps:

1. Create your own metric class and inherit ``BaseMetric`` class.
2. In the ``__init__`` function of your class, make sure to call
   ``super().__init__('name')`` where 'name' is the name of your metric. If
   you require any parameters in your ``__init__`` function, you can use
   keyword arguments to represent them and metric constructor will take care of
   providing them to your class from config.
3. Implement a ``calculate`` function which takes in ``SampleList`` and
   `model_output` as input and return back a float tensor/number.
4. Register your metric with a key 'name' by using decorator,
   ``#@ .register_metric('name')``.

Example::

    import torch

    from mmf.common.  import  
    from mmf.modules.metrics import BaseMetric

    #@ .register_metric("some")
    class SomeMetric(BaseMetric):
        def __init__(self, some_param=None):
            super().__init__("some")
            ....

        def calculate(self, sample_list, model_output):
            metric = torch.tensor(2, dtype=torch.float)
            return metric

Example config for above metric::

    model_config:
        pythia:
            metrics:
            - type: some
              params:
                some_param: a
"""

import collections

import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    recall_score,
    precision_score,
)


def _convert_to_one_hot(expected, output):
    # This won't get called in case of multilabel, only multiclass or binary
    # as multilabel will anyways be multi hot vector
    if output.squeeze().dim() != expected.squeeze().dim() and expected.dim() == 1:
        expected = torch.nn.functional.one_hot(
            expected.long(), num_classes=output.size(-1)
        ).float()
    return expected




class BaseMetric:
    """Base class to be inherited by all metrics registered to MMF. See
    the description on top of the file for more information. Child class must
    implement ``calculate`` function.

    Args:
        name (str): Name of the metric.

    """

    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.required_params = ["scores", "targets"]

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Abstract method to be implemented by the child class. Takes
        in a ``SampleList`` and a dict returned by model as output and
        returns back a float tensor/number indicating value for this metric.

        Args:
            sample_list (SampleList): SampleList provided by the dataloader for the
                                current iteration.
            model_output (Dict): Output dict from the model for the current
                                 SampleList

        Returns:
            torch.Tensor|float: Value of the metric.

        """
        # Override in your child class
        raise NotImplementedError("'calculate' must be implemented in the child class")

    def __call__(self, *args, **kwargs):
        return self.calculate(*args, **kwargs)

    def _calculate_with_checks(self, *args, **kwargs):
        value = self.calculate(*args, **kwargs)
        return value


class Accuracy(BaseMetric):
    """Metric for calculating accuracy.

    **Key:** ``accuracy``
    """

    def __init__(self):
        super().__init__("accuracy")

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate accuracy and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: accuracy.

        """
        output = model_output["scores"]
        expected = sample_list["targets"]

        assert (
            output.dim() <= 2
        ), "Output from model shouldn't have more than dim 2 for accuracy"
        assert (
            expected.dim() <= 2
        ), "Expected target shouldn't have more than dim 2 for accuracy"

        if output.dim() == 2:
            output = torch.max(output, 1)[1]

        # If more than 1
        # If last dim is 1, we directly have class indices
        if expected.dim() == 2 and expected.size(-1) != 1:
            expected = torch.max(expected, 1)[1]

        correct = (expected == output.squeeze()).sum().float()
        total = len(expected)

        value = correct / total
        return value




class VQAAccuracy(BaseMetric):
    """
    Calculate VQAAccuracy. Find more information here_

    **Key**: ``vqa_accuracy``.

    .. _here: https://visualqa.org/evaluation.html
    """

    def __init__(self):
        super().__init__("vqa_accuracy")

    def _masked_unk_softmax(self, x, dim, mask_idx):
        x1 = torch.nn.functional.softmax(x, dim=dim)
        x1[:, mask_idx] = 0
        x1_sum = torch.sum(x1, dim=1, keepdim=True)
        y = x1 / x1_sum
        return y

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate vqa accuracy and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: VQA Accuracy

        """
        output = model_output["scores"]
        # for three branch movie+mcan model
        if output.dim() == 3:
            output = output[:, 0]
        expected = sample_list["targets"]

        output = self._masked_unk_softmax(output, 1, 0)
        output = output.argmax(dim=1)  # argmax

        one_hots = expected.new_zeros(*expected.size())
        one_hots.scatter_(1, output.view(-1, 1), 1)
        scores = one_hots * expected
        accuracy = torch.sum(scores) / expected.size(0)

        return accuracy



#####################Recall
class Recall(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__("recall")
        self._multilabel = kwargs.pop("multilabel", False)
        self._sk_kwargs = kwargs

    def calculate(self, sample_list, model_output, *args, **kwargs):
        
        scores = model_output["scores"]
        expected = sample_list["targets"]

        if self._multilabel:
            output = torch.sigmoid(scores)
            output = torch.round(output)
            expected = _convert_to_one_hot(expected, output)
        else:
            # Multiclass, or binary case
            output = scores.argmax(dim=-1)
            if expected.dim() != 1:
                # Probably one-hot, convert back to class indices array
                expected = expected.argmax(dim=-1)

        value = recall_score(expected.cpu(), output.cpu(), **self._sk_kwargs)

        return expected.new_tensor(value, dtype=torch.float)
    

class BinaryRecall(Recall):

    def __init__(self, *args, **kwargs):
        super().__init__(average="micro", labels=[1], **kwargs)
        self.name = "binary_recall"
    
########################Pre

class Pre(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__("pre")
        self._multilabel = kwargs.pop("multilabel", False)
        self._sk_kwargs = kwargs

    def calculate(self, sample_list, model_output, *args, **kwargs):
        
        scores = model_output["scores"]
        expected = sample_list["targets"]

        if self._multilabel:
            output = torch.sigmoid(scores)
            output = torch.round(output)
            expected = _convert_to_one_hot(expected, output)
        else:
            # Multiclass, or binary case
            output = scores.argmax(dim=-1)
            if expected.dim() != 1:
                # Probably one-hot, convert back to class indices array
                expected = expected.argmax(dim=-1)

        value = precision_score(expected.cpu(), output.cpu(), **self._sk_kwargs)

        return expected.new_tensor(value, dtype=torch.float)


class BinaryPre(Pre):
    def __init__(self, *args, **kwargs):
        super().__init__(average="micro", labels=[1], **kwargs)
        self.name = "binary_pre"
    
###################




class F1(BaseMetric):
    """Metric for calculating F1. Can be used with type and params
    argument for customization. params will be directly passed to sklearn
    f1 function.
    **Key:** ``f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__("f1")
        self._multilabel = kwargs.pop("multilabel", False)
        self._sk_kwargs = kwargs

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate f1 and return it back.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration
            model_output (Dict): Dict returned by model.

        Returns:
            torch.FloatTensor: f1.
        """
        scores = model_output["scores"]
        expected = sample_list["targets"]

        if self._multilabel:
            output = torch.sigmoid(scores)
            output = torch.round(output)
            expected = _convert_to_one_hot(expected, output)
        else:
            # Multiclass, or binary case
            output = scores.argmax(dim=-1)
            if expected.dim() != 1:
                # Probably one-hot, convert back to class indices array
                expected = expected.argmax(dim=-1)

        value = f1_score(expected.cpu(), output.cpu(), **self._sk_kwargs)

        return expected.new_tensor(value, dtype=torch.float)


##@ .register_metric("macro_f1")
class MacroF1(F1):
    """Metric for calculating Macro F1.

    **Key:** ``macro_f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="macro", **kwargs)
        self.name = "macro_f1"


##@ .register_metric("micro_f1")
class MicroF1(F1):
    """Metric for calculating Micro F1.

    **Key:** ``micro_f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="micro", **kwargs)
        self.name = "micro_f1"


##@ .register_metric("binary_f1")
class BinaryF1(F1):
    """Metric for calculating Binary F1.

    **Key:** ``binary_f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="micro", labels=[1], **kwargs)
        self.name = "binary_f1"


##@ .register_metric("multilabel_f1")
class MultiLabelF1(F1):
    """Metric for calculating Multilabel F1.

    **Key:** ``multilabel_f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(multilabel=True, **kwargs)
        self.name = "multilabel_f1"


##@ .register_metric("multilabel_micro_f1")
class MultiLabelMicroF1(MultiLabelF1):
    """Metric for calculating Multilabel Micro F1.

    **Key:** ``multilabel_micro_f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="micro", **kwargs)
        self.name = "multilabel_micro_f1"


##@ .register_metric("multilabel_macro_f1")
class MultiLabelMacroF1(MultiLabelF1):
    """Metric for calculating Multilabel Macro F1.

    **Key:** ``multilabel_macro_f1``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="macro", **kwargs)
        self.name = "multilabel_macro_f1"


##@ .register_metric("roc_auc")
class ROC_AUC(BaseMetric):
    """Metric for calculating ROC_AUC.
    See more details at `sklearn.metrics.roc_auc_score <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html#sklearn.metrics.roc_auc_score>`_ # noqa

    **Note**: ROC_AUC is not defined when expected tensor only contains one
    label. Make sure you have both labels always or use it on full val only

    **Key:** ``roc_auc``
    """

    def __init__(self, *args, **kwargs):
        super().__init__("roc_auc")
        self._sk_kwargs = kwargs

    def calculate(self, pre, label,use_logis=True, *args, **kwargs):
        #pre:tensor [n,2]
        #label:tensor [n]
        """Calculate ROC_AUC and returns it back. The function performs softmax
        on the logits provided and then calculated the ROC_AUC.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration.
            model_output (Dict): Dict returned by model. This should contain "scores"
                                 field pointing to logits returned from the model.

        Returns:
            torch.FloatTensor: ROC_AUC.

        """
        if use_logis:
            output = torch.nn.functional.softmax(pre, dim=-1)
        else:
            output=pre
        
        expected = label
        expected = _convert_to_one_hot(expected, output)
        value = roc_auc_score(expected.cpu(), output.cpu(), **self._sk_kwargs)
        #print(len(pre),pre,output,label,expected,value)
        return expected.new_tensor(value, dtype=torch.float)


#@ .register_metric("micro_roc_auc")
class MicroROC_AUC(ROC_AUC):
    """Metric for calculating Micro ROC_AUC.

    **Key:** ``micro_roc_auc``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="micro", **kwargs)
        self.name = "micro_roc_auc"


#@ .register_metric("macro_roc_auc")
class MacroROC_AUC(ROC_AUC):
    """Metric for calculating Macro ROC_AUC.

    **Key:** ``macro_roc_auc``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="macro", **kwargs)
        self.name = "macro_roc_auc"


#@ .register_metric("ap")
class AveragePrecision(BaseMetric):
    """Metric for calculating Average Precision.
    See more details at `sklearn.metrics.average_precision_score <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score>`_ # noqa
    If you are looking for binary case, please take a look at binary_ap
    **Key:** ``ap``
    """

    def __init__(self, *args, **kwargs):
        super().__init__("ap")
        self._sk_kwargs = kwargs

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate AP and returns it back. The function performs softmax
        on the logits provided and then calculated the AP.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration.
            model_output (Dict): Dict returned by model. This should contain "scores"
                                 field pointing to logits returned from the model.

        Returns:
            torch.FloatTensor: AP.

        """

        output = torch.nn.functional.softmax(model_output["scores"], dim=-1)
        expected = sample_list["targets"]
        expected = _convert_to_one_hot(expected, output)
        value = average_precision_score(expected.cpu(), output.cpu(), **self._sk_kwargs)
        return expected.new_tensor(value, dtype=torch.float)


#@ .register_metric("binary_ap")
class BinaryAP(AveragePrecision):
    """Metric for calculating Binary Average Precision.
    See more details at `sklearn.metrics.average_precision_score <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score>`_ # noqa
    **Key:** ``binary_ap``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.name = "binary_ap"

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate Binary AP and returns it back. The function performs softmax
        on the logits provided and then calculated the binary AP.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration.
            model_output (Dict): Dict returned by model. This should contain "scores"
                                 field pointing to logits returned from the model.

        Returns:
            torch.FloatTensor: AP.

        """

        output = torch.nn.functional.softmax(model_output["scores"], dim=-1)
        # Take the score for positive (1) label
        output = output[:, 1]
        expected = sample_list["targets"]

        # One hot format -> Labels
        if expected.dim() == 2:
            expected = expected.argmax(dim=1)

        value = average_precision_score(expected.cpu(), output.cpu(), **self._sk_kwargs)
        return expected.new_tensor(value, dtype=torch.float)


#@ .register_metric("micro_ap")
class MicroAP(AveragePrecision):
    """Metric for calculating Micro Average Precision.

    **Key:** ``micro_ap``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="micro", **kwargs)
        self.name = "micro_ap"


#@ .register_metric("macro_ap")
class MacroAP(AveragePrecision):
    """Metric for calculating Macro Average Precision.

    **Key:** ``macro_ap``
    """

    def __init__(self, *args, **kwargs):
        super().__init__(average="macro", **kwargs)
        self.name = "macro_ap"


#@ .register_metric("r@pk")
class RecallAtPrecisionK(BaseMetric):
    """Metric for calculating recall when precision is above a
    particular threshold. Use `p_threshold` param to specify the
    precision threshold i.e. k. Accepts precision in both 0-1
    and 1-100 format.

    **Key:** ``r@pk``
    """

    def __init__(self, p_threshold, *args, **kwargs):
        """Initialization function recall @ precision k

        Args:
            p_threshold (float): Precision threshold
        """
        super().__init__(name="r@pk")
        self.name = "r@pk"
        self.p_threshold = p_threshold if p_threshold < 1 else p_threshold / 100

    def calculate(self, sample_list, model_output, *args, **kwargs):
        """Calculate Recall at precision k and returns it back. The function
        performs softmax on the logits provided and then calculated the metric.

        Args:
            sample_list (SampleList): SampleList provided by DataLoader for
                                current iteration.
            model_output (Dict): Dict returned by model. This should contain "scores"
                                 field pointing to logits returned from the model.

        Returns:
            torch.FloatTensor: Recall @ precision k.

        """
        output = torch.nn.functional.softmax(model_output["scores"], dim=-1)[:, 1]
        expected = sample_list["targets"]

        # One hot format -> Labels
        if expected.dim() == 2:
            expected = expected.argmax(dim=1)

        precision, recall, thresh = precision_recall_curve(expected.cpu(), output.cpu())

        try:
            value, _ = max(
                (r, p) for p, r in zip(precision, recall) if p >= self.p_threshold
            )
        except ValueError:
            value = 0

        return expected.new_tensor(value, dtype=torch.float)
