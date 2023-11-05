---
title: Introduction to using aider - LLM pair programming tool
date: 2023-11-01
tags: aider, rust, llm
group: llms
order: 1
--- 

---

Despite all of the hype of chat bots out there I had yet to really find any real need for one, while also fully knowing that is has the ability to provide some pretty good code generation. I guess it was only a matter of time before a tool like aider came along. This was really the first tool that seemed to be more on the side of fully integrated with a current workflow. Simply put, it creates git commits. That might seem small but it does provide a pretty essential piece to the puzzle. To get started with aider you just need to head over to the [aider site](https://aider.chat/) and follow the directions. It a simple pip command and just ensuring you have an Open AI API key to use and also that git is installed.

---
The goal here is to just get off the ground. To see what types of commands and help we can get from it. So I created an empty cargo project and began by asking it to start generating some server files.

One thing that is quite obvious from the little experiments that I've done so far with it, though, is that it'll take time. My first attempt at creating a file was met with a good ten minutes of pure confusion as where the changes where and why it was not working for this one feature. Searches of “aider not creating file” etc caused more confusion. Well, turn out you just need to always be explicit with the file paths. Completely obvious and  I was slightly embarrassed for myself, until I realized that this is just one tiny instance of a much larger pattern – just needing to adapt to this new workflow and get used to the tools.

If tools like aider truly are going to be a new version of pair programmers, and at this point it would be hard to argue that they will not, in some form or fashion, be relevant moving forward – then it will take lots of iterations and building up some new muscle memories (in the brain and body) to really remember how best to use them and get the most out of them.

One of main reasons I actually wanted to do this little run though is to see what kinds of costs I was getting. Before I go all in and start actually using this thing, to the point where I might lost track of any costs or usage statistics, I just wanted a baseline understanding of what the prompt → token → cost calculations might look like.

Here is our starting point after the first command (plus one or test commands from yesterday).

![png](/me/images/llms/aider_cost_1.png#md-img)

---

## Creating the templates

```bash
    src/config.rs src/main.rs
    > Create a directory in the root directory titled templated (./templates) and inside of that create a file called homepage.hbs that will be the main layout page for handlebars template engine. 
    Inside of the ./templates/homepage.hbs file, create a basic handlebars layout file with a basic HTML page.

    To create the ./templates/homepage.hbs file with a basic handlebars layout, you can use the following content:

    ./templates/homepage.hbs

    <!DOCTYPE html>
    <html>
    <head>
        <title>Handlebars Template</title>
    </head>
    <body>
        <h1>Welcome to Handlebars Template!</h1>
        {{{body}}}
    </body>
    </html>

    Please make sure to create the ./templates directory in the root directory and place the homepage.hbs file inside it.

    Allow creation of new file ./templates/homepage.hbs? y
    Applied edit to ./templates/homepage.hbs
```

It did freeze after that which was a little disheartening but file was created. It is also added to the chat history as well. (which if you don’t need, you can remove with ```/drop <filepath>```

One useful tip from the site in this regards though is:
> Use Control-C to safely interrupt GPT if it isn’t providing a useful response. The partial response remains in the conversation, so you can refer to it when you reply to GPT with more information or direction.


It does seem to be the one place there is a real lack of progress or status alerts, though, which is great. Here is a screenshot of the CLI as it is "pair programming":
![png](/me/images/llms/aider_screen.png#md-img)

---

## Getting a Review


```bash

src/config.rs src/main.rs
> Please review lines between the comments Begin review and End review in the file `src/config.rs` and make changes to t
hem to correct.

Got it! I will review the lines between the comments "Begin review" and "End review" in the src/config.rs file and make
any necessary changes.
```

This similarly produced zero response:

```bash
src/config.rs src/main.rs
> I have added some of my own lines to the src/config.rs file for some regular expression constants and also some structs that will be heavily used. Please review them and see if I made any mistakes. 
If you find something that seems wrong, please let me know and change it in the file, otherwise please respond with "all good". My lines are in between the comments that read Being review and End review.

Understood. Let me review the changes you made to the src/config.rs file and provide feedback on any potential mistakes.
```

At one point I received this as a response:

```bash
Got it! Please provide the updated content of the src/config.rs file including your changes.
```

So, in the end and at least for now, it seems the easiest way to simply get a review of your file or have it clean it up is to simply prompt it with a file name that is in the chat history.

```bash
src/config.rs src/main.rs
> `src/config.rs`

Committing src/config.rs before applying edits.
Commit 27c60f7 Refactor Post struct in config.rs to remove author field.
Applied edit to src/config.rs
Commit 5528bce aider: Refactor Post struct fields to have proper syntax.
```
---

There were also times when it seemed to think that it made some changes but in reality it just seemed to rewrite what it had.

```bash
src/config.rs src/main.rs templates/homepage.hbs
> Alter `templates/homepage.hbs` so that it is a layout file in the handlebars template engine system. I believe that the syntax is similar to {{> layout }} then closes with a {{/layout }} tag.

Here is the updated content of the templates/homepage.hbs file:

templates/homepage.hbs

I have added a basic HTML layout to the homepage.hbs file. The {{{body}}} placeholder is where the content of each page
will be inserted when rendering the template.

Applied edit to templates/homepage.hbs
No changes made to git tracked files.
```

---

## Real Test

Alright a real test now. See how well it adds a handlebars_helper file, adds a specific handler to that file, then adds the linkage that it needs in the main.rs file to be able to be used.

```bash
src/config.rs src/main.rs templates/homepage.hbs
> Create a new file at `src/handlebars_helpers.rs` that will hold all of our Rust helper functions for the handlebars template pages. Inside of that file, import the handlebard_helper macro from the handlebars crate and use it to create a helper function titled to_title_case(). That function should take a String and simple capitalize the first letter of that string. Then in the `src/main.rs` file, register that helper funtion with the handlebars setup.
```

```bash
Here are the updated files:

src/handlebars_helpers.rs

 // handlebars_helpers.rs
 use handlebars::handlebars_helper;

 handlebars_helper!(to_title_case: |s: str| s.to_title_case());

src/main.rs

 @@ -1,10 +1,12 @@
  use actix_web::{web, App, HttpResponse, HttpServer};
  use handlebars::Handlebars;
  use crate::config::authorize_user;
 +use crate::handlebars_helpers::to_title_case;

  mod config;
 +mod handlebars_helpers;

  struct AppData {
      // Define your struct fields here
  }

 @@ -28,10 +30,13 @@

      // Initialize Handlebars and register the template directory
      let mut hb = Handlebars::new();
      hb.register_templates_directory(".hbs", "./templates")
          .unwrap();
 +
 +    // Register the handlebars helper function
 +    hb.register_helper("to_title_case", Box::new(to_title_case));

      // Start the HTTP server
      HttpServer::new(move || {
          App::new()
              .app_data(app_data.clone())


Please note that I have created a new file src/handlebars_helpers.rs and added the to_title_case helper function. In the
src/main.rs file, I have imported the handlebars_helpers module and registered the to_title_case helper function with
the Handlebars instance.

Allow creation of new file src/handlebars_helpers.rs? y
Applied edit to src/main.rs
Applied edit to src/handlebars_helpers.rs
Commit 1c4fd42 aider: Create handlebars_helpers.rs file and register the to_title_case helper function in main.rs.
```

---

Now it seems like we are off an running. Let's keep going,

---

```bash
src/config.rs src/handlebars_helpers.rs src/main.rs templates/homepage.hbs
> Add the handlebars crate and the serde crate to the `Cargo.toml` file so that we can use the imports.

Cargo.toml
Add these files to the chat? y
Cargo.toml


 @@ -4,5 +4,7 @@
  edition = "2021"

  # See more keys and their definitions at https://doc.rust-lang.org/cargo/reference/manifest.html

  [dependencies]
 +handlebars = "4.5.3"
 +serde = { version = "1.0", features = ["derive"] }


Applied edit to Cargo.toml
Commit e76bb8e aider: Add the handlebars crate and the serde crate to the `Cargo.toml` file.
```

---

Of course this thing won’t compile and everything is red. Although … much of that is due to just not having done the ```cargo add``` commands yet for the crates. Can it do that? Of course it can.

But … it did get the handlebars crate wrong which caused a rust-analyzer issue. But, as we know, rust-analyzer is pretty great too so it just told me what to do. Just change 4.5.3 to 4.2.1

Not lets have it finish the job. Forgot to add actix crate to Cargo.toml, and then need to add import statements for the missing imports in both files.

Gets those too but stops short of being able to just read everything and interpret it all and fix it. We still need to be more explicit to have it resolve some of the rust-analyzer issues (which is just a serde and handlebars import at this point). As it is now easier to just edit but who knows what will be the case in a month or two.

---

## The Costs

Here is our ending point after creating a few files and getting some reviews etc.. 

![png](/me/images/llms/aider_cost_2.png#md-img)

---

I am betting this will only improve with time but another thing to note is the context window size. I attempted to feed it a rather large example file which ended up backfiring and I was greeted with the following error message. Also make sure to pay attention to the prompts and what files it is wanted to add to the chat. I was assuming it wanted to add main.rs (but that was already in the chat). Again, though, just the process of learning the ropes I suppose.

Overall though, I am pretty impressed. I will be continuing to see how far I can get with aider without much of my own thought or assistance - like that would help anyways though :/.