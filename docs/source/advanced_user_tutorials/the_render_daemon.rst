=================
The Render Daemon
=================

When you first run an Algan problem, there are some overhead costs assosciated with
starting up the renderer. Firstly, Algan's depencies (mainly Torch, Taichi) must
be imported and initialized. Secondly, Taichi must compile all of the rendering kernels.
All up, this takes about 20 seconds give or take. To reduce the start-up time
and make quickly iterating on a scene more conveniant, Algan therefore employs
a render daemon. The render daemon is simply a copy of Algan, running on a
different process. When you import Algan into a script, Algan will look
for a daemon which is already running and, if there is one, run
the script on the existing daemon instead. If no daemon is found,
Algan automatically launches one.
This way, the start-up cost is only paid once, on the daemon, and every subsequent
script run will immediately begin rendering.
The daemon process has an idle timeout of 2 hours, if no script uses
it in that Window it will be killed, and the next Algan script
will launch a new one, paying the start-up cost agian.
