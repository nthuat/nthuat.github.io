---
layout: post
title: Thread lifecycle in Java
---

As an Android developer, you have to work directly or indirectly with threads in your apps. So understanding Thread Lifecycle in Java very important. 

## Lifecycle
This following diagram illustrates the states of thread:

<p align="center">
<img src="/assets/thread_lifecycle_java.jpeg"/>
</p>


### New

Before the execution thread, a Thread object is created. At this point, the thread is not alive. It remains this state until the program starts the thread by calling `Thread.start`.

### Runnable
When we call `Thread.start`, the thread's state is changed to Runnable. The execution environment is set up and the thread is ready to be executed. The control is given to thread scheduler to finish its execution. Whether to run this thread instantly or keep it in runnable thread pool before running, depends on the implementation of thread scheduler.

### Running
Thread scheduler picks one of the thread from the runnable thread pool and change its state to Running. The `run` method is called and the task is executed.

### Blocked
A thread can enter Blocked state due to waiting for the resources that are held by another thread.
* Waiting for I/O resources
* Waiting for a monitor lock

### Waiting
A thread is in the waiting state due to calling one of the following methods:
* `Object.wait` with no timeout
* `Thread.join` with no timeout

A thread in the waiting state is waiting for another thread to perform a particular action. For example, a thread that has called `Object.wait` on an object is waiting for another thread to call `Object.notify` or `Object.notifyAll` on that object.
Once the thread wait state is over, it's state is changed to Runnable and it's moved back to runnable thread pool.

### Timed_Waiting
A thread is in the timed waiting state due to calling one of the following methods with a specified positive waiting time:
* `Thread.sleep`
* `Object.wait` with timeout
* `Thread.join` with timeout

Once the thread wait state is over, its state is changed to Runnable and it's moved back to runnable thread pool.

### Terminated
When the `run` method has finished execution, the thread is terminated and its resources can be freed up. This is the final state of the thread and no reuse of the thread instance or its execution environment is possible.

## Wrap up
This article explains the states a thread can enter during its existence. I will write more interesting stuff related to threads in the next articles. Happy reading!

## References
- <https://docs.oracle.com/javase/8/docs/api/java/lang/Thread.State.html>
- <https://www.baeldung.com/java-thread-lifecycle>