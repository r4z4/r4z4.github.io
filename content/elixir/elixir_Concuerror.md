---
title: Concurrency Testing w/ Concuerror
date: 2023-08-07
tags: elixir, erlang, testing, concurrency
group: elixir
order: 3
--- 

---
![png](/me/images/elixir/Concuerror.png#md-img-bright)

---

Continuing with the theme of just being different, for better or worse, Erlang also gives us a good opportunity to explore two other, more obscure in my experience, methods of testing in Property Based Testing and also Concurrency Testing. I had no prior experience with either and they were both rather interesting to look into.

---

Property Based testing really originated with Haskell and the QuickCheck implementation and seems that most other implementations in other languages, including Erlang, just follow this approach. The difficulty that I have had here is just finding what I would need to test in this manner. It is just a little different way of thinking about the testing so it does not fit well for all cases.

---


Concurrency testing was also completely new to me, but I certainly understood the need and importance of this. But unlike property based testing, it was the actual implementation and internals of how it works that evaded me. 
I fully intended this to kind of be a one off and just to try it, but I actually think it might offer quite a but as I go and add more features to the system. If you think about it, the whole reason to choose to use this whole ecosystem is for the concurrency offerings so this is just another one of those tools that can help. It is just a matter of getting used to it and learning the new terms etc.

---

On that note, the [Concuerror](https://www.concuerror.com) site does have some pretty good tutorials which are helpful to walk through some of the initial stages. Here they describe a few of the terms and values to look out for in a typical report:

---

---

We need to set up our environment before we can use the tool though, of course. I have not had great experiences using some of the Erlang tooling, but I was actually pleasently surprised here and found the process a little more tolerable, which was nice. Even with that said, though, there is still some configuration to do. First, need to make sure the right files get compiled. Concuerror does not work with .exs files, so we need to make sure that we only compile the tests that we need and that we
can avoid the extentions. That is done by altering the mix.exs configuration file to read:

```elixir
def project do
    [
      # ...
      elixirc_paths: elixirc_paths(Mix.env),
      test_pattern: "*_test.ex*",
      warn_test_pattern: nil
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/concuerror"]
  defp elixirc_paths(_), do: ["lib"]
  ```

It is also worth noting that if you will want to have the images for easy viewing, it helps to install graphviz, which will enable you to use the ```dot``` command, which we will use like this after we run out test and it generates the .dot file for us, which we will the convert to a .png image.

```shell-session
dot -Tpng my_graph.dot > my_graph.png
```

Now our images are availalbe. But of course we are still early in our testing and still getting failures. I'll just use one of the images but trust me I had many of these. After some trial and error runs and then finally giving up and reading some documentation, turns out there are a few little intricacies that we need to account for first. Here is the fhat first failing image.

--- 
![png](/me/images/elixir/subscription_graph.png#md-img)

---

This is a version of the first concurrency test I was running, with only one interleaving, but even so I was running into some issues, all seeming to do with just making sure that we have our ```handle_cast``` or ```handle_info``` callbacks properly set up in our module.

```elixir
defmodule FanCan.ConcurrencyTest do

  def push(pid, n) do
      GenServer.cast(pid, {:new_message, n})
  end

  @doc """
  For now this is just a simple test. Need to perform some actions
  with the elements and continue to test
  """
  def test do
    # {:ok, pid} = GenServer.start_link(FanCanWeb.ThreadLive.Index, [])
    {:ok, pid} = GenServer.start_link(FanCanWeb.SubscriptionServer, [])

    ["a","b","c","d","e","f","g"] |> Enum.each(fn ltr -> push(pid, ltr) end)

    ["h","i","j","k","l","m","n"] |> Enum.each(fn ltr -> push(pid, ltr) end)

    ["o","p","q","r","s","t","u"] |> Enum.each(fn ltr -> push(pid, ltr) end)

    ["u","v","w","x","y","z","!"] |> Enum.each(fn ltr -> push(pid, ltr) end)

    # This is necessary. Even if there was no crash of the GenServer,
    # not stopping it would make Concuerror believe that the process
    # is stuck forever, as no new process can ever send it messages in
    # this test.
    GenServer.stop(pid)
  end
end
```

And now here is the SubscriptionServer module and the implementation of the GenServer callback methods.


```elixir
defmodule FanCanWeb.SubscriptionServer do
  use GenServer
  alias FanCan.Core.TopicHelpers
  alias FanCan.Accounts.UserFollows
  
  def start do
    initial_state = []
    receive_messages(initial_state)
  end

  def receive_messages(state) do
    receive do
      msg ->
        {:ok, new_state} = handle_message(msg, state)
        receive_messages(new_state)
    end
  end

  def handle_message({:subscribe_user_follows, user_follows}, state) do
    for follow = %UserFollows{} <- user_follows do
      IO.inspect(follow, label: "Type")
      # Subscribe to user_follows. E.g. forums that user subscribes to
      case follow.type do
        :candidate -> TopicHelpers.subscribe_to_followers("candidate", follow.follow_ids)
        :user -> TopicHelpers.subscribe_to_followers("user", follow.follow_ids)
        :forum -> TopicHelpers.subscribe_to_followers("forum", follow.follow_ids)
        :election -> TopicHelpers.subscribe_to_followers("election", follow.follow_ids)
      end
    end
    {:ok, []}
  end

  def handle_message({:subscribe_user_published, current_user_published_ids}, state) do
    with %{post_ids: post_ids, thread_ids: thread_ids} <- current_user_published_ids do
      IO.inspect(thread_ids, label: "thread_ids_b")
      for post_id <- post_ids do
        FanCanWeb.Endpoint.subscribe("posts_" <> post_id)
      end
      for thread_id <- thread_ids do
        FanCanWeb.Endpoint.subscribe("threads_" <> thread_id)
      end
    end
    {:ok, []}
  end

  def handle_info(_) do
    IO.puts "Info Handler"
    {:ok, []}
  end

  def handle_cast({:new_message, element}, state) do
    IO.inspect(state, label: "Concuerror - State var")
    IO.inspect(element, label: "Concuerror - Element var")
    if String.contains?(element, "b") do
      String.upcase(element)
    end
    {:noreply, [element | state]}
  end

```

Most of my errors centered around getting those arguments correct, which took some getting used to tracking them down in the Concuerror logs. Here are some examples from the logs that will give us some idea of what we can look for when things go wrong.

---

There are also a few little steps which you might run into as well depending on how your system in configured. E.g I needed to update my path w/ ```export PATH=/Concuerror/bin:$PATH``` - which is where my instance of Concuerror was installed to. There were also some permissions to account for if you end up utilizing the above script in some fasion. These are pretty easily solved with a ```chmod +x ./concuerror_test.sh``` or similar.

---

Here was the final command that I ended up needing to run in order to satisfy all the tests. The hints are there if you squint your eyes and look hard enough in the docs, but it ceratinly took me a little while to find out the different options needed.

```shell-session
  ./concuerror_test.sh FanCan.ConcurrencyTest --graph my_graph.dot --after_timeout 1000 --treat_as_normal shutdown
```

After all of that, though, we are finally able to get a running test and see some green.

![png](/me/images/elixir/success_graph.png#md-img)


Similarly, it is much nicer to look at the report of the successful test versus the one that is riddled with errors. Here is the complete .txt file that is generated.


```shell-session
Concuerror 0.21.0+build.2371.refaf91d78 started at 12 Jul 2023 17:58:22
 Options:
  [{after_timeout,1000},
   {assertions_only,false},
   {assume_racing,true},
   {depth_bound,500},
   {disable_sleep_sets,false},
   {dpor,optimal},
   {entry_point,{'Elixir.FanCan.ConcurrencyTest',test,[]}},
   {exclude_module,[]},
   {first_process_errors_only,false},
   {ignore_error,[]},
   {instant_delivery,true},
   {interleaving_bound,infinity},
   {keep_going,true},
   {log_all,false},
   {non_racing_system,[]},
   {pa,"/usr/local/lib/elixir/lib/elixir/ebin"},
   {pa,"/usr/local/lib/elixir/lib/ex_unit/ebin"},
   {pa,"_build/test/lib/fan_can/ebin/"},
   {print_depth,20},
   {scheduling,round_robin},
   {scheduling_bound_type,none},
   {show_races,false},
   {strict_scheduling,false},
   {symbolic_names,true},
   {timeout,5000},
   {treat_as_normal,[shutdown]},
   {use_receive_patterns,true}]
################################################################################
Exploration completed!
  No errors found!
################################################################################
Tips:
--------------------------------------------------------------------------------
* Check `--help attributes' for info on how to pass options via module attributes.
* Running without a scheduling_bound corresponds to verification and may take a long time.
* Increase '--print_depth' if output/graph contains "...".
* Your test sends messages to the 'user' process to write output. Such messages from different processes may race, 
* producing spurious interleavings. Consider using '--non_racing_system user' to avoid them.

################################################################################
Info:
--------------------------------------------------------------------------------
* Showing progress ('-h progress', for details)
* Writing results in concuerror_report.txt
* Writing graph in my_graph.dot
* Only logging errors ('--log_all false')
* Automatically instrumented module io_lib
* Showing PIDs as "<symbolic name(/last registered name)>" ('-h symbolic_names').
* Automatically instrumented module error_handler
* Automatically instrumented module 'Elixir.FanCan.ConcurrencyTest'
* Automatically instrumented module 'Elixir.GenServer'
* Automatically instrumented module 'Elixir.Keyword'
* Automatically instrumented module gen
* Automatically instrumented module proc_lib
* Automatically instrumented module gen_server
* Automatically instrumented module 'Elixir.FanCanWeb.SubscriptionServer'
* Automatically instrumented module 'Elixir.Enum'
* Automatically instrumented module lists
* Automatically instrumented module sys
* Automatically instrumented module 'Elixir.IO'
* Automatically instrumented module io

################################################################################
Done at 12 Jul 2023 17:58:26 (Exit status: ok)
  Summary: 0 errors, 1/1 interleavings explored
```



