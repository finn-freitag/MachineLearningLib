using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib
{
    public delegate void Parallelizer(int inclusiveFrom, int exclusiveTo, Action<int> action);
}
