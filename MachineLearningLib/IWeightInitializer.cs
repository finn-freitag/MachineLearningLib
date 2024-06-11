using MachineLearningLib.NeuralNetwork;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib
{
    public interface IWeightInitializer
    {
        /// <summary>
        /// This method would be called to specify the layer for which weight should be generated. This method is for informational purposes. It should not generate weights!
        /// </summary>
        void SetLayer(Layer layer);
        float GetInitialWeight();
    }
}
