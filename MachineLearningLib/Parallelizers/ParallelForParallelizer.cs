﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachineLearningLib.Parallelizers
{
    public class ParallelForParallelizer : IParallelizer
    {
        public void Parallelizer(int from, int to, Action<int> action) => Parallel.For(from, to, action);
    }
}
