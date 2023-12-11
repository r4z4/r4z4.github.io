---
title: Basic Concurrency in Elixir
date: 2023-12-08
tags: elixir, concurrency, genserver
group: elixir
order: 1
--- 

---

After having spent some time away from Elixir for a little bit and then returning to it, I found myself a little lost on certain things at first. It does differ just enough in some of the little things it feels like. With that said I felt it’d be a good idea to write up a little something to show the general benefits and ways of doing things in Elixir for my own benefit, should I find myself in a similar situation in the future, which I very much anticipate. Also, though, it seems many are still not quite yet familiar with some of those benefits, so a quick and concise little showcase might be a nice little thing to be able to point too as well. [You can view it here.](https://stream-handler-render-db.onrender.com/)

---

So, with that said, it might make sense to describe the project. I just wanted to showcase how some of those basic concurrency primitives like spawn, send & receive can be used in implementations like  GenServers to help you build highly-concurrent applications. The simplest thing I could think of was a dashboard-like UI with some buttons that’ll launch and stop some asynchronous services. We’ll also utilize Phoenix LiveView which’ll allow all of those updates to be automatically sent to the UI via the websocket connection.

---

Even though the application is relatively simple, there are many steps that I found myself having to pause and consult some old materials or sample code to remember what it was I needed to do in certain situations. Some of the larger ideas like GenServers or Tasks are easy to remember and hang onto, but some of the little things like how to send the equivalent of an onClick() call when a button is clicked or how to define the structs.

---

The good news here is that our simple application design will really just allow me to show the whole end-to-end flow for one of those services, and the rest is pretty much just that duplicated over and over, just being plugged into different services. All of them just utilize a GenServer process to complete their work and send back messages to the parent process (our LiveView) at each interval.

---

Makes sense to start from the very beginning though. Navigating from home page to new page.  Sounds dumb but I stumbled here. Not proud to admit it, but … In my defense though it also was a bit of the overall template-like nature of LiveView and how the views interact with the controllers. 

---

```elixir
<.link href={~p"/stream"}>Stream Page</.link>
``` 
& then it clicked (no pun intended) that there was the router.ex module where we set all of these up.

```elixir
scope "/", StreamHandlerWeb do
  pipe_through :browser

  get "/", PageController, :home

  live "/stream", StreamLive.Index, :index
  live "/stream/new", StreamLive.Index, :new
  live "/stream/:id/edit", StreamLive.Index, :edit

  live "/stream/:id", StreamLive.Show, :show
  live "/stream/:id/show/edit", StreamLive.Show, :edit
end
```

---

The only route I am even using here is root and ```/stream```, but just left the others for show. As you can see, when he hit the ```/stream``` route, that will be handled by the StreamLive.Index module. Since we have that new page now we can design a basic layout to get things started.

```html
<.header>
  Listing
</.header>
  <div>
    <div></div>
    <div class="h-56 border-black border-4 p-8">
      <button>Launch Service</button>
    </div>
  </div>
```
---

Just starting with a button that will be used to launch a service and an area to display the result. With that we can take a look at the StreamLive.Index module which will house all of the logic that will dictate what gets displayed here. This is really where things started to flood back to me. That mostly occurred because I saw this:

```elixir
def mount(_params, _session, socket) do
  IO.puts “Mounted”

  {:ok,
    socket
    |> assign(:state, %{})
  }
end
```

---

Then you start to remember it all about the lifecycle & the assigns. I'm not going to delve into the basic of those as I believe [others do a much better job at that](https://blog.appsignal.com/2022/06/14/a-guide-to-phoenix-liveview-assigns.html). I’ll touch briefly on each of the variations here just so that none of the concepts are undefined, but all in all they all work in a similar manner, which makes our job much easier. 

---

#### APIs

Perhaps the most common use case is simply reaching out to some third party service to get some data and then working with that in some manner, to again send it back via the GenServer. This is the pattern followed in the Emojis, Slugs and Activities cards but I decided I would show another version of one that I ended up not using, only because it includes the extra API key param and header, which you can simply omit if the API does not require them (as none of those others do). Here I am simply just sending that response right back after some decoding.

```elixir
defp get_aq() do
  {:ok, resp} =
    Finch.build(
      :get,
      "https://api.openaq.org/v1/locations?limit=100&page=1&offset=0&sort=desc&radius=1000&city=ARAPAHOE&order_by=lastUpdated&dump_raw=false",
      [{"Accept", "application/json"}, {"X-API-Key", System.fetch_env!("OPEN_AQ_KEY")}]
      )
      |> Finch.request(StreamHandler.Finch)

  IO.inspect(resp, label: "Resp")
  {:ok, body} = Jason.decode(resp.body)
  body
end
```

So with our function written that will return us our data, we can set up the rest of the GenServer. This will be the flow that each of the other services will follow (some with slight variations).
Our service is started when we click the start button, which casts a message to our GenServer. As you can see, later on the same thing happens for stopping the resource


```elixir
"3" ->
  IO.puts "Activities Casted"
  GenServer.cast :consumer_4, {:fetch_resource, :activities}
"12" ->
  IO.puts "Activities Stopped"
  GenServer.cast :consumer_4, {:stop_resource, :activities}
```

So now we can look in the Producer module (what our ```:consumer_4``` is) to see how that ```:fetch_resource``` message is being handled. Since it is a casted message, it will be handled by the ```handle_cast/2``` method.


```elixir
@impl true
def handle_cast({:fetch_resource, sym}, state) do
  Process.send_after(self(), sym, @time_interval_ms)
  IO.puts(sym)
  {:noreply, state}
end
```

```handle_cast/2``` will handle all our the external messages that get casted to the GenServer. When it receives one of those that matches our message, it then sends itself another message after a certain interval, which will be handled by a different method, ```handle_info/2```. Since Producer module has many different services, we need to find the one that matches the ```sym``` variable, which here is ```:activities```.

```elixir
@impl true
def handle_info(:activities, state) do
  activities_ref = Process.send_after(self(), :activities, @call_interval_ms)
  body = get_activities()
  activity =
    case Map.fetch(body, "activity") do
      {:ok, str} -> str
      :error -> "No Activity"
    end
  type =
    case Map.fetch(body, "type") do
      {:ok, str} -> str
      :error -> "No Type"
    end
  display = %{activity: activity, type: type}
  Phoenix.PubSub.broadcast(
    StreamHandler.PubSub,
    @activities,
    %{topic: @activities, payload: %{status: :complete, data: display, text: "Activities has completed."}}
  )
  state = Map.put(state, :activities_ref, activities_ref)
  {:noreply, state}
end
```
The message is received, we call our ```get_activities/0``` function which returns a body of activities data. We need to manipulate that a tiny bit to clean it up, and then we can send it through the [PubSub](https://hexdocs.pm/phoenix_pubsub/Phoenix.PubSub.html) mechanism where we have subscribed to each process. In a very similar manner, this is a message that gets passed, pattern matched on in the LiveView module - ```Index.ex``` - and then displayed.

```elixir
@impl true
def handle_info(%{topic: @activities, payload: msg}, socket) do
  IO.inspect(socket)
  IO.inspect(msg, label: "Msg")
  IO.puts "Handle Broadcast for Activities"
  {:noreply,
    socket
    |> assign(:activities, msg[:data])
  }
end
```

That assigns at the end is what will be picked up in our ```index.html.heex``` template file, like so:

```html
<div class={get_class(@clicked_map, 3)}>
  <StreamHandlerWeb.DashboardComponents.activities_card activities={@activities} />
</div>
```

... which is just one div in our grid of nine.
That pretty much does it. Each of those @interval values that we set a new message will be sent and the process will repeat itself. Now we can look at some of the other pieces and how they may differ ever so slightly.

---

#### ETS

Erlang Term Storage is an in-memory storage option similar to Redis for those familiar. Similar to Redis, it provides solutions to things like caching and easy data access, which are among two of the more common use cases. Here, we will simply be using it a hold our user_score data to resemble a leaderboard. Our initial values will come from the DB, and then they will be updated each round (to simulate some game play) and then the scores will be recalculated, resorted and send back via the GenServer so that we can display it however. For this one, we are using a simple HTML table. 

If this were a production scenario, we would need to design some synchronization mechanisms to ensure that the scores are being persisted in the database in a timely manner.


```elixir
def get_user_scores do
  # ms = :ets.fun2ms fn {score, username, joined} -> {score, username, joined} end
  tuples = :ets.tab2list(:user_scores)
  user_scores =
    Enum.map(tuples, fn {username, score, joined} ->
      %UserScore{username: username, score: score, joined: joined}
    end)
  user_scores
end
```

In a similar manner, then, we relay that through. 

```elixir
@impl true
def handle_info(:ets, state) do
  IO.puts "Handle ETS"
  scores = get_user_scores()
  shuffle_user_scores(scores)
  sorted = Enum.sort_by(scores, &(&1.score), :desc)
  publish_str(@ets, sorted)
  ets_ref = Process.send_after(self(), :ets, @time_interval_ms)
  state = Map.put(state, :ets_ref, ets_ref)
  {:noreply, state}
end
```

One additional step to note here is the ets_ref that we assign to the send_after call. This ref is then saved to our GenServer state, and upon the call to :stop_resouce, that ref is found and used to initiate a call to ```Process.cancel_timer/2```, which will stop the service, until it is restarted.

```elixir
@impl true
def handle_cast({:stop_resource, sym}, state) do
  IO.inspect(state, label: "State")
  case sym do
    :images -> Process.cancel_timer(state.images_ref)
    :reader -> Process.cancel_timer(state.reader_ref)
    :ets -> Process.cancel_timer(state.ets_ref)
    _ -> Process.cancel_timer(state.reader_ref)
  end
  IO.puts(sym)
  {:noreply, state}
end
```

If this were a production scenario, we would need to design some synchronization mechanisms to ensure that the scores are being persisted in the database in a timely manner.

---

StreamData
StreamData is a library that provides two very useful features for Elixir apps: data generation and property testing. While I was primarily interested in the data generation for the services task, I did briefly introduce a property test using the library. 

```elixir
property "struct! macro correctly translates UserScore struct" do
  check all username <- binary(),
            score <- integer(),
            joined <- naive_dt_generator() do
    assert struct!(UserScore, %{username: username, score: score, joined: joined}) == %UserScore{username: username, score: score, joined: joined}
  end
end
```

For the data generation side, though, the process is much more straightforward. For this particular service I simply want to create some arbitrary data so that we can send it back via the GenServer and then react to that data, in this case by plotting the points on a map. So while our data here is completely generated, the process would be the exact same to just swap in whatever backend processes you might have.

```elixir
def generate_data do
  # Get 10 lists of random # of tuples
  list_of_lists =
    StreamData.list_of(
      StreamData.tuple({
        StreamData.float(min: 0, max: @max),
        StreamData.float(min: 0, max: @max_2)}
      )
    ) |> Enum.take(10)
  combined_list = Enum.reduce(list_of_lists, fn x, acc -> x ++ acc end)
  combined_list
end
```

---

#### WebSockex
[WebSockex](https://hexdocs.pm/websockex/WebSockex.html) is a popular Elixir websocket client, which is an implementation of a GenServer which makes our job very similar to the ones proceeding it. Given that it is just another GenServer, we are able to use the same callback methods to interact with it. If you go to the WebSockex docs you can see that there are some slight variants and additions, but for the most part it is still the same ```handle_info/2``` and ```handle_cast/2``` that we are used to.

But, since it is a websocket client, ```handle_call/2``` is replaced with ```handle_frame/2```.


```elixir
@impl true
def handle_frame({:text, data}, state) do
  Logger.info("Received: #{inspect(data)}")
  new_message = Jason.decode!(data)
  IO.inspect(new_message, label: "NEW MESSAGE HEY")
  case new_message do
    %{"event" => "heartbeat"} -> IO.puts("Heartbeat")
    %{"connectionID" => conn, "event" => event, "status" => status, "version" => version} -> IO.puts("Connection")
    new_message when map_size(new_message) == 6 -> IO.puts("Other 6 Connection")
    _ -> broadcast(new_message)
  end
  {:ok, state}
end
[...]
def broadcast(msg) do
  IO.inspect(msg, label: "Broadcast MSG")
  case Kernel.elem(List.pop_at(msg, 2), 0) do
    "spread" -> StreamHandlerWeb.Endpoint.broadcast!("websocket", "new_spread", msg)
    "book-10" -> StreamHandlerWeb.Endpoint.broadcast!("websocket", "new_message", msg)
    # Just subscribing to the ticker endpoint
    # "ticker" -> StreamHandlerWeb.Endpoint.broadcast!("websocket", "new_ticker", msg)
    _ -> IO.puts("Ugh")
  end
end
```

I won't go into the details on the actual data structures here as they are not really relevant, but essentially of these subscriptions returns a JSON structure with some fields indicting prices and bids for each resource.

---


#### File Read
This one is certainly nothing special. I just wanted to include it because it is such a common task and eventually you will probably have to do something similar, so it is nice to know it works in much the same way.

---


#### Observer/Phoenix Dashboard

Despite the simplicity of our services it does still result in a lot of activity, which led me to remember about using erlang observer to take a look at running systems. Turns out there are some extra steps to be able to connect to that the running Iex session, but luckily Phoenix ships with a feature called LiveDashboard which basically takes all the :observer information make it available through LiveView route. You can also add your own metrics to customize it. I have not gotten that far yet though so I will just show you some basic screenshots of what it gives us just out of the box.

If you do want to use :observer, all you need to do is add it to the extra_applications list, along with :wx. and :os_mon. You can leave off :os_mon if uninterested in OS metrics.

```elixir
def application do
  [
    mod: {StreamHandler.Application, []},
    extra_applications: [:logger, :runtime_tools, :wx, :observer, :os_mon]
  ]
end
```

Along with the standard overview/dashboard UI and metrics that you get, you can also look at each individual process.

![png](/images/elixir/pdash_process.png#md-img-vert)

This is certainly one area that I plan to explore some more as I continue, but it does show some of the nice built-in tools that are available.

---

All in all this was a very low effort project that resulted in some pretty good, yet simple, concurrent behavior. I plan to keep scaling this up to see how things keep performing, so hopefully I can follow up soon with some better APIs and maybe some GenServers that actually do something!

[If you would like to, you can view the source code here](https://github.com/r4z4/stream_handler)

Thanks for reading!