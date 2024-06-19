using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib
{
    public interface IWeightInitializable
    {
        IWeightInitializer WeightInitializer { get; set; }
    }
}
