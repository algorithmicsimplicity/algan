Currently, each Mob is handled as an independent Python object, with its own data dictionary
and modification history. This induces a loooot of overhead for scenes with many thousands of Mobs,
significantly slowing down animation.
So, we want to rework the Mob system to use batch processing. This is a significant architectural change,
will require rewriting much of Mob and Scene.

# The basic idea is:

## At the very beginning of running a script, during Algan import, pre-allocate a ManualMemory buffer tensor of several GB.
## Whenever a new Mob is created, it allocates room in the global buffer for all of its animated attributes (location,
basis, color, normal, material properties, etc, etc), and writes its data to its allocated slot in the global buffer.
## Each Mob stores the indexes into the global buffer where its data is stored.
## When a Mob is made a child of another, the parent will add the child's indexes to its collection of descendant indexes.
## When an animation is applied to a Mob, such as mob.move(UP), the mob gathers all of its descendants indexes (for attribute
location) (including its own index for location), and performs a batched update to the global buffer targeting
all of the descendant indexes (including itself) in one operation.
## The mob then records in a GLOBAL modification history the time the modification took place, the set of indices
affected, and the set of values.
## The animated_func is then recorded to a GLOBAL modification history.
## At animation time, the pre-animated-function global state for time T is produced in one batched operation, then
all animated functions which take place during the given time T are executed with interpolated params to produce
the final global state at time T. Since modifications to animated attributes now update all descendants in one operation,
the animation functions should be quick to apply at animation time, no need to loop through the entire mob hierarchy
of Python objects.
## The produced global state vector is then immediately shipped to GPU for rendering.

# Things to be aware of:
## In order to efficiently compute pre-animated-function state, we need a function that takes the global
modification history and a range of time_inds as input and constructs the state at those times. In order to do this
efficiently we need to represent the set of modifications in an appropriate data structure (sorted by time of modification),
then for each index in the global memory buffer do a binary search (bisect) to find the modification that is
relevant to it. This will probably need to be implemented as its own dedicated Taichi kernel to make it efficient
(even if it is done on CPU, the Python overhead will be brutal. So the Taichi kernel will help for CPU execution as well).
## In order for animated attribute modifications to be efficient we need to make sure as much as possible that descendants'
attributes are stored next to each other, so that when a parent does a write operation to all of its descendants, they
are all contiguous in global buffer memory. At the same time it's probably infeasible to ensure that it is always
 a contiguous chunk (like when a triangle is the parent of bezier), so the write function will probably need to accept as input a list of (start, end) ranges,
 and write to the contiguous chunks between each start and end range. Again, this might need its own dedicated Taichi kernel.
## The rendering function accepts data in the form of a Primitive for each primitive type (flat triangle, PN triangle, Bezier circuit),
so if possible, at animation time when materializing state, try to materialize in 3 separate chunks one for each primitive type,
so they can immediately be wrapped in the Primitive classes and sent to render without needing to separate out the primitives
and rewrite them as contiguous chunks.
## When a mob is set as child of another, you must update the parent's stored set of descendant indexes and then recurse
to further ancestors as well.
