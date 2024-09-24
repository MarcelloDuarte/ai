<?php

namespace spec\Md\Ai;

use Md\Ai\Perceptron;
use PhpSpec\ObjectBehavior;

class PerceptronSpec extends ObjectBehavior
{
    function it_is_initializable()
    {
        $this->shouldHaveType(Perceptron::class);
    }

    function it_has_weights_initialized_to_zero()
    {
        $this->getWeights()->shouldBe([0, 0, 0]);
    }

    function it_calculates_the_weighted_sum_of_inputs_and_weights()
    {
        $this->setWeights([0.5, 0.3, 0.2]);
        $inputs = [1, 2, 3];

        $sum = $this->calculateWeightedSum($inputs)->getWrappedObject();

        if (abs($sum - 1.7) > 0.00001) {
            $this->fail("Expected sum to be approximately 1.7, but got {$sum}");
        }
    }

    function it_updates_weights_based_on_learning_rate_and_error()
    {
        $this->setWeights([0.5, -0.4, 0.2]);
        $inputs = [1, 0, 1];
        $learningRate = 0.1;
        $desiredOutput = 1;
        $weightedSum = $this->calculateWeightedSum($inputs);
        $actualOutput = $this->activate($weightedSum->getWrappedObject())->getWrappedObject();
        $error = $desiredOutput - $actualOutput;
        $this->updateWeights($inputs, $learningRate, $error);
        $expectedWeights = [
            0.5 + $learningRate * $error * $inputs[0],
            -0.4 + $learningRate * $error * $inputs[1],
            0.2 + $learningRate * $error * $inputs[2],
        ];
        $this->getWeights()->shouldBe($expectedWeights);
    }

    function it_activates_the_perceptron_based_on_weighted_sum()
    {
        $this->setWeights([0.5, -0.4, 0.2]);
        $inputs = [1, 0, 1];
        $weightedSum = $this->calculateWeightedSum($inputs);

        $activationResult = $this->activate($weightedSum->getWrappedObject())->getWrappedObject();

        if ($activationResult !== 1) {
            $this->fail("Expected activation result to be 1, but got {$activationResult}");
        }
    }

    function it_updates_bias_based_on_learning_rate_and_error()
    {
        $this->setBias(0.5);
        $learningRate = 0.1;
        $desiredOutput = 1;
        $actualOutput = 0;
        $error = $desiredOutput - $actualOutput;
        $this->updateBias($learningRate, $error);
        $expectedBias = 0.5 + $learningRate * $error;
        $this->getBias()->shouldBe($expectedBias);
    }

    function it_correctly_classifies_training_data_after_training()
    {
        $this->setWeights([0.5, -0.4, 0.2]);
        $inputs = [
            [1, 0, 1], // Expected output: 1
            [0, 1, 1], // Expected output: 1
            [1, 1, 0], // Expected output: 0
        ];
        $desiredOutputs = [1, 1, 0];
        $learningRate = 0.1;

        // Training Loop
        for ($epoch = 0; $epoch < 100; $epoch++) {
            foreach ($inputs as $index => $input) {
                $weightedSum = $this->calculateWeightedSum($input)->getWrappedObject();
                $actualOutput = $this->activate($weightedSum)->getWrappedObject();
                $error = $desiredOutputs[$index] - $actualOutput;
                $this->updateWeights($input, $learningRate, $error);
            }
        }

        // Verification Loop
        foreach ($inputs as $index => $input) {
            $weightedSum = $this->calculateWeightedSum($input)->getWrappedObject();
            $actualOutput = $this->activate($weightedSum)->getWrappedObject();
            $desiredOutput = $desiredOutputs[$index];

            if ($actualOutput !== $desiredOutput) {
                throw new \Exception(
                    "After training, perceptron output for input " .
                    "[" . implode(', ', $input) . "] was {$actualOutput}, expected {$desiredOutput}."
                );
            }
        }
    }
}
