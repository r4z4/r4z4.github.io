---
title: Benchmarking in Elixir Using Benchee (& Erlang)
date: 2023-08-01
tags: elixir, benchmark, erlang
group: elixir
order: 1
--- 

---

I never really thought that I’d get to the point of wanting or needing to profile my operations, but after you spend a lot of time doing the same thing over and over, it is natural to want to try and find the best way to do it. From a pure efficiency perspective, it makes sense to save x nanoseconds each time we operate if we know we will be operating many thousands of times per day 😊.

Plus, it is just the responsible thing to do. I cannot rebel against growing up in every facet of my life. 

---

This first really came up when realizing the extent of my Enum and List operations. I always knew “List bad for access” but its just so natural and hard to shake … things just go in lists. So, let’s put some hard numbers on it to finally, once and for all, convince myself to go the other way. Tuple vs. List. One comparison that I found along the way was the think of a tuple as a DB row. Typically each of the elements relate to a common source, and you can see why you would not want to enumerate over that.

---

It is worth noting we get suspicious results when just utilizing the built-in Erlang tools. To be fair, though, they do provide a warning for us in the [OTP Efficiency Guide](https://www.erlang.org/doc/efficiency_guide/profiling.html#benchmarking).

>timer:tc/3 measures wall-clock time. The advantage with wall-clock time is that I/O, swapping, and other activities in the operating system kernel are included in the measurements. The disadvantage is that the measurements vary a lot. Usually it is best to run the benchmark several times and note the shortest time, which is to be the minimum time that is possible to achieve under the best of circumstances ... Therefore, measuring CPU time is misleading if any I/O (file or socket) is involved.


---
Here is the signature for the ```tc/2``` function (the additional argument in the ```tc/3``` function is the TimeUnit, which will be defaulted to microseconds when we use the ```tc/2``` function)

```tc(Fun, Arguments) -> {Time, Value}``` which means we will be using the ```tc/2``` function.

When we run this function just within our application, though, the results are not so great.

```erlang
{uSecs, :ok} = :timer.tc(&func/arity, [func-args])
```

```shell-session
List Time: 0
Tup Time: 1
```

Of course this is also likely a factor of just having a much smaller function as well, as we will see here soon when we run the Erlang timer on the much more expensive operation. There are also other modules w/in Erlang you can use too, like ```erlang::statistics```.
With that said, we'll turn to [Benchee](https://github.com/bencheeorg/benchee). I did use another Library called Benchfella to begin but Benchee is actively supported & maintained and seems to be the one that most Elixir
developers will reach for.  
  

Gotta do the usual to add it to the project. Unlike Benchfella though we do not need to directly add it as a child process in our main application function.

```elixir
defp deps do
  [{:benchee, "~> 1.0", only: :dev}]
end
```

Then do ...

```elixir
$ mix deps.get
...
$ mix compile
```

Now just set up a file and run it. You can either create a module as you normally might for a custom module you want to implement, or we can just create a script and then run it with a ```mix run``` command. We'll use the second option here just because it will be much simpler to execute and go in and change to our needs. If I needed to build upon them as the application evolved, a formal module might be the preferred approach. Create the file ```bencnmark.exs```.

```elixir
list = Enum.to_list(1..10_000)
tuple = List.to_tuple(list)

Benchee.run(
  %{
    "EnumAt" => fn -> Enum.at(list, 2222) end,
    "KernelElem" => fn -> Kernel.elem(tuple, 2222) end
  },
  time: 10,
  memory_time: 2
)
```

```shell-session
Benchmarking EnumAt ...
Benchmarking KernelElem ...
[notice]     :alarm_handler: {:set, {:system_memory_high_watermark, []}}
[notice]     :alarm_handler: {:set, {:process_memory_high_watermark, #PID<0.101.0>}}

Name                 ips        average  deviation         median         99th %
KernelElem       25.49 M      0.0392 μs ±21713.36%           0 μs       0.100 μs
EnumAt           0.143 M        7.01 μs  ±1104.59%        5.90 μs       21.20 μs

Comparison: 
KernelElem       25.49 M
EnumAt           0.143 M - 178.67x slower +6.97 μs

Memory usage statistics:

Name          Memory usage
KernelElem             0 B
EnumAt                16 B - ∞ x memory usage +16 B

**All measurements for memory usage were the same**
```

Using the libraries helps us isolate the functions and eliminate a lot of the noise that can contribute to faulty or suspicious results.

```shell-session
List Time: 88
Tup Time: 0
```





