const extrevDesc = 'A PERN stack app mimicing commercial approaches to case management, in this case healthcare appeals. PostgreSQL for database, Node.js and Express.js' + 
  ' as backend and served via React.js. App enables authorization via a third party service, Keycloak, although currently disabled to allow unauthorized access.';
const sarpyDesc = 'A simple data visualization project for a local housing authority agency.  Created with ReCharts library using housing data provided by the agency.';
const cityDesc = 'A simple City Explorer app to fetch weather data and forecasts and Teleport city information for urban areas.  Utilizes two RESTful API' +
  'services and serves up via ReactJS on the frontend. ****NOTE: Troposphere has ended their API services as of 1/2022 - Using a temporary API until I can find a better one. Open to suggestions :) ****';
const frontpageDesc = 'A Newspaper Front Page generator app. SPA done with React.js. Still looking for the Batman newspaper audio clip.';
const triviaDesc = 'A simple trivia game.  Test your knowledge on topics: Movies, Geography, Literature, History, Science-Technology, Animals and Sports.' +
  'Made with Vue.js, Express.js and PostgreSQL.';
const elmMonsterDesc = 'A more appropriate version of the hangman game written in Elm. It really is a delightful front end language.';
const scrambleDesc = 'A typing exercise/brain teaser written in Elm. Working on getting a better API.';
const quickUIDesc = 'A simple UI for any group or organization.  Sub in any remote or local data to display by details and sort via tags.  Created with Vue.js.';

document.addEventListener("DOMContentLoaded", function() {  //ES6 version of $(document).ready

  /*>>>>>>>>>>>>>>>>>>>>>>>>App Buttons Hide/Show and Attaching Event Handlers<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<*/

//TO ES6 NOTES ==> .toggleClass is now .classList.toggle // $ is now document.querySelector // Removed .stop(true) but did not replace it - do I need to?

document.getElementById("cta").addEventListener("click", function() {

  document.getElementById("fwHeader1").classList.contains("fade-out")
  ? document.getElementById("fwHeader1").classList.replace("fade-out", "fade-in") 
  : document.getElementById("fwHeader1").classList.add("fade-in");

  document.getElementById("fwHeader2").classList.contains("fade-out")
  ? document.getElementById("fwHeader2").classList.replace("fade-out", "fade-in") 
  : document.getElementById("fwHeader2").classList.add("fade-in");

  document.getElementById("fwHeader3").classList.contains("fade-out")
  ? document.getElementById("fwHeader3").classList.replace("fade-out", "fade-in") 
  : document.getElementById("fwHeader3").classList.add("fade-in");

  document.getElementById("fwImg1").classList.contains("fade-out")
  ? document.getElementById("fwImg1").classList.replace("fade-out", "fade-in") 
  : document.getElementById("fwImg1").classList.add("fade-in");

  document.getElementById("fwImg2").classList.contains("fade-out")
  ? document.getElementById("fwImg2").classList.replace("fade-out", "fade-in") 
  : document.getElementById("fwImg2").classList.add("fade-in");

  document.getElementById("fwImg3").classList.contains("fade-out")
  ? document.getElementById("fwImg3").classList.replace("fade-out", "fade-in") 
  : document.getElementById("fwImg3").classList.add("fade-in");

  document.getElementById("cta_previews_pern").classList.contains("fade-out")
  ? document.getElementById("cta_previews_pern").classList.replace("fade-out", "fade-in") 
  : document.getElementById("cta_previews_pern").classList.add("fade-in");

  document.getElementById("cta_previews_monster").classList.contains("fade-out")
  ? document.getElementById("cta_previews_monster").classList.replace("fade-out", "fade-in") 
  : document.getElementById("cta_previews_monster").classList.add("fade-in");

  document.getElementById("cta_previews_scramble").classList.contains("fade-out")
  ? document.getElementById("cta_previews_scramble").classList.replace("fade-out", "fade-in") 
  : document.getElementById("cta_previews_scramble").classList.add("fade-in");

  document.getElementById("cta_previews_fp").classList.contains("fade-out")
  ? document.getElementById("cta_previews_fp").classList.replace("fade-out", "fade-in") 
  : document.getElementById("cta_previews_fp").classList.add("fade-in");

  document.getElementById("cta_previews_sarpy").classList.contains("fade-out")
  ? document.getElementById("cta_previews_sarpy").classList.replace("fade-out", "fade-in") 
  : document.getElementById("cta_previews_sarpy").classList.add("fade-in");

  document.getElementById("cta_previews_api").classList.contains("fade-out")
  ? document.getElementById("cta_previews_api").classList.replace("fade-out", "fade-in") 
  : document.getElementById("cta_previews_api").classList.add("fade-in");

  document.getElementById("cta_previews_trivia").classList.contains("fade-out")
  ? document.getElementById("cta_previews_trivia").classList.replace("fade-out", "fade-in") 
  : document.getElementById("cta_previews_trivia").classList.add("fade-in");

  document.getElementById("cta_previews_quick").classList.contains("fade-out")
  ? document.getElementById("cta_previews_quick").classList.replace("fade-out", "fade-in") 
  : document.getElementById("cta_previews_quick").classList.add("fade-in");

  }, false); 

document.getElementById( "cta_wipe" ).addEventListener("click", function(e) {

  document.getElementById("fwHeader1").classList.contains("fade-in") 
  ? document.getElementById("fwHeader1").classList.replace("fade-in", "fade-out") 
  : document.getElementById("fwHeader1").classList.add("fade-out");

  document.getElementById("fwHeader2").classList.contains("fade-in") 
  ? document.getElementById("fwHeader2").classList.replace("fade-in", "fade-out") 
  : document.getElementById("fwHeader2").classList.add("fade-out");

  document.getElementById("fwHeader3").classList.contains("fade-in") 
  ? document.getElementById("fwHeader3").classList.replace("fade-in", "fade-out") 
  : document.getElementById("fwHeader3").classList.add("fade-out");

  document.getElementById("fwImg1").classList.contains("fade-in") 
  ? document.getElementById("fwImg1").classList.replace("fade-in", "fade-out") 
  : document.getElementById("fwImg1").classList.add("fade-out");

  document.getElementById("fwImg2").classList.contains("fade-in") 
  ? document.getElementById("fwImg2").classList.replace("fade-in", "fade-out") 
  : document.getElementById("fwImg2").classList.add("fade-out");

  document.getElementById("fwImg3").classList.contains("fade-in") 
  ? document.getElementById("fwImg3").classList.replace("fade-in", "fade-out") 
  : document.getElementById("fwImg3").classList.add("fade-out");
  
  document.getElementById("cta_previews_pern").classList.contains("fade-in") 
  ? document.getElementById("cta_previews_pern").classList.replace("fade-in", "fade-out") 
  : document.getElementById("cta_previews_pern").classList.add("fade-out");

  document.getElementById("cta_previews_monster").classList.contains("fade-in") 
  ? document.getElementById("cta_previews_monster").classList.replace("fade-in", "fade-out") 
  : document.getElementById("cta_previews_monster").classList.add("fade-out");

  document.getElementById("cta_previews_scramble").classList.contains("fade-in") 
  ? document.getElementById("cta_previews_scramble").classList.replace("fade-in", "fade-out") 
  : document.getElementById("cta_previews_scramble").classList.add("fade-out");

    document.getElementById("cta_previews_fp").classList.contains("fade-in") 
  ? document.getElementById("cta_previews_fp").classList.replace("fade-in", "fade-out") 
  : document.getElementById("cta_previews_fp").classList.add("fade-out");

  document.getElementById("cta_previews_sarpy").classList.contains("fade-in") 
  ? document.getElementById("cta_previews_sarpy").classList.replace("fade-in", "fade-out") 
  : document.getElementById("cta_previews_sarpy").classList.add("fade-out");

  document.getElementById("cta_previews_api").classList.contains("fade-in") 
  ? document.getElementById("cta_previews_api").classList.replace("fade-in", "fade-out") 
  : document.getElementById("cta_previews_api").classList.add("fade-out");

  document.getElementById("cta_previews_trivia").classList.contains("fade-in") 
  ? document.getElementById("cta_previews_trivia").classList.replace("fade-in", "fade-out") 
  : document.getElementById("cta_previews_trivia").classList.add("fade-out");

  document.getElementById("cta_previews_quick").classList.contains("fade-in") 
  ? document.getElementById("cta_previews_quick").classList.replace("fade-in", "fade-out") 
  : document.getElementById("cta_previews_quick").classList.add("fade-out");

  }, false); 

  /*>>>>>>>>>>>>>>>>>>>>>>>>>Each Button Handlers tp Display Respective Preview<<<<<<<<<<<<<<<<<<<<<<<<<<<<<*/

  function addImage(filepath) { 
    var img = document.createElement('img'); 
    img.src = filepath; 
    document.getElementById('sitePreview').appendChild(img);
  }

  function addDesc(desc) { 
    document.getElementById('sitePreviewDesc').innerHTML = desc;
  }

  document.getElementById("cta_previews_pern").addEventListener("mouseenter", function() {
    addImage('../images/extrevPERN.PNG');
    addDesc(extrevDesc);
    }, false); 
    
  document.getElementById("cta_previews_pern").addEventListener("mouseleave", function() {
    document.querySelector("#sitePreview").innerHTML = null;
    document.querySelector("#sitePreviewDesc").innerHTML = null;
    }, false);

    document.getElementById("cta_previews_pern").addEventListener("click", function() {
    window.open('https://pern-extrev.herokuapp.com/', '_blank');
  });

  document.getElementById("cta_previews_monster").addEventListener("mouseenter", function() {
    addImage('../images/elmMonsterGame.PNG');
    addDesc(elmMonsterDesc);
    }, false); 
    
  document.getElementById("cta_previews_monster").addEventListener("mouseleave", function() {
    document.querySelector("#sitePreview").innerHTML = null;
    document.querySelector("#sitePreviewDesc").innerHTML = null;
    }, false);

    document.getElementById("cta_previews_monster").addEventListener("click", function() {
    window.open('https://objective-keller-8aec52.netlify.app/', '_blank');
  });

  document.getElementById("cta_previews_scramble").addEventListener("mouseenter", function() {
    addImage('../images/scramble.PNG');
    addDesc(scrambleDesc);
    }, false); 
    
  document.getElementById("cta_previews_scramble").addEventListener("mouseleave", function() {
    document.querySelector("#sitePreview").innerHTML = null;
    document.querySelector("#sitePreviewDesc").innerHTML = null;
    }, false);

    document.getElementById("cta_previews_scramble").addEventListener("click", function() {
    window.open('https://epic-easley-ff5cfd.netlify.app/', '_blank');
  });

  document.getElementById("cta_previews_sarpy").addEventListener("mouseenter", function() {
    addImage('../images/sarpy.PNG');
    addDesc(sarpyDesc);
    }, false); 
    
  document.getElementById("cta_previews_sarpy").addEventListener("mouseleave", function() {
    document.querySelector("#sitePreview").innerHTML = null;
    document.querySelector("#sitePreviewDesc").innerHTML = null;
    }, false);

    document.getElementById("cta_previews_sarpy").addEventListener("click", function() {
    window.open('https://sarpy-county.vercel.app/', '_blank');
  });

  document.getElementById("cta_previews_api").addEventListener("mouseenter", function() {
    addImage('../images/cityAPIweather.PNG');
    addDesc(cityDesc);
    }, false); 
    
  document.getElementById("cta_previews_api").addEventListener("mouseleave", function() {
    document.querySelector("#sitePreview").innerHTML = null;
    document.querySelector("#sitePreviewDesc").innerHTML = null;
    }, false);

    document.getElementById("cta_previews_api").addEventListener("click", function() {
    window.open('https://city-explorer.vercel.app/', '_blank');
  });

  document.getElementById("cta_previews_fp").addEventListener("mouseover", function() {
    addImage('../images/fp3.PNG');
    addDesc(frontpageDesc);
    }, false); 

  document.getElementById("cta_previews_fp").addEventListener("mouseleave", function() {
    document.querySelector("#sitePreview").innerHTML = null;
    document.querySelector("#sitePreviewDesc").innerHTML = null;
    }, false);

    document.getElementById("cta_previews_fp").addEventListener("click", function() {
    window.open('https://front-page-sigma.vercel.app/', '_blank');
  });

  document.getElementById("cta_previews_quick").addEventListener("mouseover", function() {
    addImage('../images/quickUI.PNG');
    addDesc(quickUIDesc);
    }, false); 

  document.getElementById("cta_previews_quick").addEventListener("mouseleave", function() {
    document.querySelector("#sitePreview").innerHTML = null;
    document.querySelector("#sitePreviewDesc").innerHTML = null;
    }, false);

    document.getElementById("cta_previews_quick").addEventListener("click", function() {
    window.open('https://elastic-lamport-5e120f.netlify.app/', '_blank');
  });

  document.getElementById("cta_previews_trivia").addEventListener("mouseover", function() {
    addImage('../images/trivia.PNG');
    addDesc(triviaDesc);
    }, false); 

  document.getElementById("cta_previews_trivia").addEventListener("mouseleave", function() {
    document.querySelector("#sitePreview").innerHTML = null;
    document.querySelector("#sitePreviewDesc").innerHTML = null;
    }, false);

    document.getElementById("cta_previews_trivia").addEventListener("click", function() {
    window.open('https://simple-trivia-v.herokuapp.com/', '_blank');
  });

 

}); // End Ready

var buttons = document.querySelectorAll(".toggle-button");
var modal = document.querySelector("#modal");

[].forEach.call(buttons, function (button) {
  button.addEventListener("click", function () {
    modal.classList.toggle("off");
  });
});
