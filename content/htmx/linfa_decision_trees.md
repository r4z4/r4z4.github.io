---
title: Decision Trees in Rust using the Linfa crate
date: 2023-11-30
tags: linfa, rust, ml
group: htmx
order: 2
--- 

---

A little while back I came across the [linfa crate](https://github.com/rust-ml/linfa), which is one of the more data science libraries in the Rust ecosystem. Since then I have been toying around with it a bit, and the combination of the speed and the type safety made it a very pleasing experience to the point where I really needed to start to use it more in some of my projects.

---


Before getting started it might be worth a very brief intro to Linfa, which is probably most easily done via the similarities to NumPy per the Rust docs...
>   * Arrays have a single element type.
>   * Arrays can have arbitrarily many dimensions.
>   * Arrays can have arbitrary strides.
>   * Indexing starts at zero, not one.
>   * The default memory layout is row-major, and the default iterators follow row-major order (also called “logical order” in the documentation).
>   * Arithmetic operators work elementwise. (For example, a * b performs elementwise multiplication, not matrix multiplication.)
>   * Owned arrays are contiguous in memory.
>   * Many operations, such as slicing, are very cheap because they can return a view of an array instead of copying the data.


---

Of course there are some differences that will come up, mostly dealing with the Rust type system and its handling of generics with the ArrayBase type, as well as some slight variations on the methodology of the slice() method. None of these play a direct role in the decision trees we are creating here. Now, onto our example.

---

Admittedly this is one of those solutions finding a problem scenarios, but I was able to generate a little workflow that utilizes the linfa-trees crate to compose a Decision Tree model from current database data, and then make a prediction on user submitted form data which is sent in to represent a new event (in this case it is a meeting/consult).

---

The basic idea is what we would assume we are setting something up to where we want the model to predict the user (here it is a consultant type) to assign to the meeting. We can fix up our data to have it predict on the optimal case, and then we can use the model’s prediction for that model data to get our desired output. In terms of the actual value that we get from this particular case is really nothing to write home about, but it certainly illustrates the power of linfa, and the ease of introducing it into your projects.

---

With all that said, the first step in all of this is to get our current data, stored in our Postgres DB, into the form that we need for the Dataset. Here is our code to get the data from the database, and convert it into an Array2 struct that linfa can work with:

---

```rust
async fn build_model_ndarray(db: &Pool<Postgres>) -> Result<Array2<f32>, String> {
    match sqlx::query_as::<_, ModelData>(
        "SELECT consult_purpose_id, client_type_id, consults.client_id, clients.specialty_id, clients.territory_id, location_id, notes, consult_result_id, num_attendees, consult_start, consult_end
                FROM consults INNER JOIN clients ON consults.client_id = clients.id WHERE consult_end < now()",
    )
    .fetch_all(db)
    .await
    {
        Ok(model_data) => {
            let built_arr: Array2<f32> = model_data.iter()
                .map(|row| row.as_f32array())
                .collect::<Vec<_>>()
                .into();
            Ok(built_arr)
        },
        Err(e) => Err(format!("Error in DB {}", e).to_string())
    }
}
```

---

From there, one of the key pieces that we need to implement ourselves is to make sure that we can convert all of this data into the ```Array2<f32>``` that linfa needs. This whole step is really where the type safety (& rust-analyzer) is there as a nice backstop. 

---

```rust
impl LinfaPredictionInput {
    pub fn as_ndarray(&self) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 2]>> {
        array!([
            self.consult_purpose_id as f32,
            self.client_id as f32,
            self.client_type as f32,
            self.specialty_id as f32,
            self.territory_id as f32,
            self.location_id as f32,
            self.notes_length as f32,
            self.meeting_duration as f32,
            self.hour_of_day as f32,
            self.received_follow_up as f32,
            self.num_attendees as f32,
        ])
    }
}
```

---

We can then put these pieces together into the main linda_predict function that will do the actual prediction.

---

```rust
pub async fn linfa_pred(input: &LinfaPredictionInput, pool: &Pool<Postgres>) -> LinfaPredictionResult {
let built_arr: Array2<f32> = build_model_ndarray(pool).await.unwrap();
    dbg!(&built_arr);

    let feature_names = vec![
        "consult_purpose_id",
        "client_type",
        "client_id",
        "specialty_id",
        "territory_id",
        "location_id",
        "notes_length",
        "meeting_duration",
        "hour_of_day",
        "received_follow_up",
        "num_attendees",
        "consultant_id",
    ];
    let num_features = built_arr.len_of(Axis(1)) - 1;
    let features = built_arr.slice(s![.., 0..num_features]).to_owned();
    let labels = built_arr.column(num_features).to_owned();

    let linfa_dataset = Dataset::new(features, labels)
        .map_targets(|x| match x.to_owned() as i32 {
            1 => "Hulk Hogan",
            2 => "Mike",
            3 => "Zardos",
            4 => "Greg",
            5 => "Rob",
            6 => "Vanessa",
            7 => "Joe",
            _ => "Nobody",
        })
        .with_feature_names(feature_names);

    let model = DecisionTree::params()
        .split_quality(SplitQuality::Gini)
        .fit(&linfa_dataset)
        .unwrap();

    let input: Array2<f32> = input.as_ndarray();
    let predictions = model.predict(&input);

    // Map back to int. FIXME
    let pred = predictions[0];
    let consultant_id =
        match pred {
            "Hulk Hogan" => 1,
            "Mike" => 2,
            "Zardos" => 3,
            "Greg" => 4,
            "Rob" => 5,
            "Vanessa" => 6,
            "Joe" => 7,
            _ => 0,
        };

    // Create Decision Tree file for each generation for audit/review/records. FIXME: Export to Storage (GCP)
    let path = "./static/linfa/consults/";
    let filename = Uuid::new_v4().to_string();
    let ext = ".tex";
    File::create(format!("{}{}{}", path, filename, ext))
        .unwrap()
        .write_all(model.export_to_tikz().with_legend().to_string().as_bytes())
        .unwrap();

    // return tuple struct (filename_uuid, consultant_id)
    LinfaPredictionResult(filename, consultant_id)
}
```

---

If we focus in on the last section here, this is one more critical pieces to the reason why the ability to utilize this crate easily into your projects is so nice. If you have ever worked in an industry or setting where there is a strong need for evidence-based reasoning or proof-of-work, a leading document to be able to point back to is just something that many other algorithms and libraries can offer you. Decision tree libraries will likely always have this available of course, but when it combined with the power of Rust and the ease of use is really when things started to click.

---

From here, it is simple to add the ability to generate a LaTeX document for the model that made the prediction. We can then generate a UUID for that file and save it back to the database to associate with the record that we just created. 


```rust
    File::create(format!("{}{}{}", path, filename, ext))
        .unwrap()
        .write_all(model.export_to_tikz().with_legend().to_string().as_bytes())
        .unwrap();
```

---
Here is an example of our TKV file, which can be converted to a PDF file pretty simply using readily available tools depending on your operating system. For myself operating in WSL it was as simple as:

```bash
sudo apt-get install texlive
pdflatex /path/to/myfile.tex
```

I have cut off the entire left side of the tree here for brevity and for ease of viewing. The left half is largely the same just for the rest of the variables that you see in the legend.

![png](/images/rust/linfa_decision_tree.png#md-img-hor)

---

This is great for environments that require more of an explainable-AI-like approach. It does feel that the more that the LLM models progress, we get further away from the principals championed by the explainable-AI field.
Our little toy example here, if it were to be applied, also deals with people and work assignments. It is easy to imagine a scenario where feelings might get hurt, work may be effected and even lives can change. Having a tree to walk back up to evaluate the decisions at each point would be a great tool to lean on in these cases where evidence-based models are required or warranted.

---

As a final note, this is still a rather raw version that can still use a little clean up. In particular I wanted to keep the map_targets() function in to show an easy way to put the result back in the same terms as the input, but of course that leads to the re-mapping that we need to do a little later on. For the sake of the application and how it is now, though, returning the name provides a better user experience. We will simply need to move this mapping elsewhere in the future. Maybe that will be the next post, although I am still diving into the linfa docs and seeing if there are some other algorithms that we can employ here with our toy scenarios.