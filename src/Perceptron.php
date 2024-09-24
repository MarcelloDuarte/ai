<?php

namespace Md\Ai;

class Perceptron
{
    private array $weights = [0, 0, 0];
    private float $bias;

    public function getWeights(): array
    {
        return $this->weights;
    }

    public function setWeights(array $weights): void
    {
        $this->weights = $weights;
    }

    public function calculateWeightedSum(array $inputs): float
    {
        $sum = 0;
        foreach ($this->weights as $index => $weight) {
            $sum += $weight * $inputs[$index];
        }

        return $sum;
    }

    public function activate($weightedSum): int
    {
        return $weightedSum >= 0 ? 1 : 0;
    }

    public function updateWeights(array $inputs, float $learningRate, float $error): void
    {
        foreach ($this->weights as $index => &$weight) {
            $weight += $learningRate * $error * $inputs[$index];
        }
    }

    public function setBias($bias): void
    {
        $this->bias = $bias;
    }

    public function updateBias(float $learningRate, float $error): void
    {
        $this->bias = $this->bias + $learningRate * $error;
    }

    public function getBias(): float
    {
        return $this->bias;
    }
}
