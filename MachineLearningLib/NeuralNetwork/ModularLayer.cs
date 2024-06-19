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
    public class ModularLayer : Layer, IActivatable, IWeightInitializable, IAccelerable, IParallelizable, IUtilizer
    {
        public IActivationFunction ActivationFunction { get; set; } = new SigmoidActivation();
        public IWeightInitializer WeightInitializer { get; set; } = new RandomWeightInitializer();
        public IAccelerator Accelerator { get; set; } = new NoAccelerator();
        public Parallelizer Parallelizer { get; set; } = NoParallelizer.Parallelizer;

        public ModularLayer(int neurons) : base(neurons)
        {
        }

        public override void InitFromPreviousLayer()
        {
            WeightInitializer.SetLayer(this);
            Weights = new float[NeuronsSum.Length][];
            Parallelizer(0, NeuronsSum.Length, (i) =>
            {
                Weights[i] = new float[PreviousLayer.NeuronsSum.Length];
                Biases[i] = WeightInitializer.GetInitialWeight();
                for (int j = 0; j < PreviousLayer.NeuronsSum.Length; j++)
                {
                    Weights[i][j] = WeightInitializer.GetInitialWeight();
                }
            });
        }

        public override void Calculate()
        {
            Parallelizer(0, NeuronsSum.Length, (i) =>
            {
                NeuronsSum[i] = Accelerator.DotProduct(PreviousLayer.NeuronsAF, Weights[i]);
                NeuronsSum[i] += Biases[i];
                NeuronsAF[i] = ActivationFunction.Evaluate(NeuronsSum[i]);
            });
            FollowingLayer.Calculate();
        }

        public override void Train(float learningRate)
        {
            Parallelizer(0, NeuronsSum.Length, (i) =>
            {
                Errors[i] = Accelerator.DotProductT(FollowingLayer.Errors, FollowingLayer.Weights, i);
                Errors[i] *= ActivationFunction.Derivative(NeuronsSum[i]);
                Weights[i] = Accelerator.Add(Weights[i], Accelerator.Multiply(learningRate * Errors[i], PreviousLayer.NeuronsAF));
                Biases[i] += learningRate * Errors[i];
            });
            PreviousLayer.Train(learningRate);
        }

        public virtual void Use(object usable)
        {
            if (usable is IActivationFunction act)
                ActivationFunction = act;
            if(usable is IWeightInitializer wei)
                WeightInitializer = wei;
            if (usable is IAccelerator acc)
                Accelerator = acc;
            if (usable is Parallelizer par)
                Parallelizer = par;
        }
    }
}
