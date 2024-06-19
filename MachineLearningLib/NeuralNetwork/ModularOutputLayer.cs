using MachineLearningLib.Accelerators;
using MachineLearningLib.ActivationFunctions;
using MachineLearningLib.Parallelizers;
using MachineLearningLib.WeightInitializers;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib.NeuralNetwork
{
    public class ModularOutputLayer : OutputLayer, IActivatable, IWeightInitializable, IAccelerable, IParallelizable, IUtilizer
    {
        public IActivationFunction ActivationFunction { get; set; } = new SigmoidActivation();
        public IWeightInitializer WeightInitializer { get; set; } = new RandomWeightInitializer();
        public IAccelerator Accelerator { get; set; } = new NoAccelerator();
        public Parallelizer Parallelizer { get; set; } = NoParallelizer.Parallelizer;

        public ModularOutputLayer(int neurons) : base(neurons)
        {
        }

        public override void Calculate()
        {
            Parallelizer(0, NeuronsSum.Length, (i) =>
            {
                NeuronsSum[i] = Accelerator.DotProduct(PreviousLayer.NeuronsAF, Weights[i]);
                NeuronsSum[i] += Biases[i];
                NeuronsAF[i] = ActivationFunction.Evaluate(NeuronsSum[i]);
            });
        }

        public override void Train(float learningRate)
        {
            Parallelizer(0, NeuronsSum.Length, (i) =>
            {
                Errors[i] = (desiredOutputs[i] - NeuronsAF[i]) * ActivationFunction.Derivative(NeuronsSum[i]);
                Weights[i] = Accelerator.Add(Weights[i], Accelerator.Multiply(learningRate * Errors[i], PreviousLayer.NeuronsAF));
                Biases[i] += learningRate * Errors[i];
            });
            PreviousLayer.Train(learningRate);
        }

        public virtual void Use(object usable)
        {
            if (usable is IActivationFunction act)
                ActivationFunction = act;
            if (usable is IWeightInitializer wei)
                WeightInitializer = wei;
            if (usable is IAccelerator acc)
                Accelerator = acc;
            if (usable is Parallelizer par)
                Parallelizer = par;
        }
    }
}
