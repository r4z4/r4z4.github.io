const extrevDesc = 'A PERN stack app mimicing commercial approaches to case management, in this case healthcare appeals. PostgreSQL for database, Node.js and Express.js' + 
  'as backend and served via React.js. App enables authorization via a third party service, Keycloak, although currently disabled to allow unauthorized access.';
const sarpyDesc = 'A simple data visualization project for a local housing authority agency.  Created with ReCharts library using housing data provided by the agency.';
const cityDesc = 'A simple City Explorer app to fetch weather data and forecasts and Teleport city information for urban areas.  Utilizes two RESTful API' +
  'services and serves up via ReactJS on the frontend. ****** (Teleport City API has ended their service - In the process of finding new public API)'
const frontpageDesc = 'A Newspaper Front Page generator app. Single SPA done with React.js. Still looking for the Batman newspaper audio clip.';
const triviaDesc = 'A simple trivia game.  Test your knowledge on topics: Movies, Geography, Literature, History, Science-Technology, Animals and Sports.' +
  'Made with Vue.js, Express.js and PostgreSQL.'

document.addEventListener("DOMContentLoaded", function() {  //ES6 version of $(document).ready

  /*>>>>>>>>>>>>>>>>>>>>>>>>App Buttons Hide/Show and Attaching Event Handlers<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<*/

//TO ES6 NOTES ==> .toggleClass is now .classList.toggle // $ is now document.querySelector // Removed .stop(true) but did not replace it - do I need to?

document.querySelector(".cta").addEventListener("click", function() {

  document.querySelector(".cta_previews_pern").classList.contains("fade-out")
  ? document.querySelector(".cta_previews_pern").classList.replace("fade-out", "fade-in") 
  : document.querySelector(".cta_previews_pern").classList.add("fade-in");

  document.querySelector(".cta_previews_fp").classList.contains("fade-out")
  ? document.querySelector(".cta_previews_fp").classList.replace("fade-out", "fade-in") 
  : document.querySelector(".cta_previews_fp").classList.add("fade-in");

  document.querySelector(".cta_previews_sarpy").classList.contains("fade-out")
  ? document.querySelector(".cta_previews_sarpy").classList.replace("fade-out", "fade-in") 
  : document.querySelector(".cta_previews_sarpy").classList.add("fade-in");

  document.querySelector(".cta_previews_api").classList.contains("fade-out")
  ? document.querySelector(".cta_previews_api").classList.replace("fade-out", "fade-in") 
  : document.querySelector(".cta_previews_api").classList.add("fade-in");

  document.querySelector(".cta_previews_trivia").classList.contains("fade-out")
  ? document.querySelector(".cta_previews_trivia").classList.replace("fade-out", "fade-in") 
  : document.querySelector(".cta_previews_trivia").classList.add("fade-in");

  }, false); 

document.querySelector( ".cta_wipe" ).addEventListener("click", function(e) {
  
  document.querySelector(".cta_previews_pern").classList.contains("fade-in") 
  ? document.querySelector(".cta_previews_pern").classList.replace("fade-in", "fade-out") 
  : document.querySelector(".cta_previews_pern").classList.add("fade-out");

    document.querySelector(".cta_previews_fp").classList.contains("fade-in") 
  ? document.querySelector(".cta_previews_fp").classList.replace("fade-in", "fade-out") 
  : document.querySelector(".cta_previews_fp").classList.add("fade-out");

  document.querySelector(".cta_previews_sarpy").classList.contains("fade-in") 
  ? document.querySelector(".cta_previews_sarpy").classList.replace("fade-in", "fade-out") 
  : document.querySelector(".cta_previews_sarpy").classList.add("fade-out");

  document.querySelector(".cta_previews_api").classList.contains("fade-in") 
  ? document.querySelector(".cta_previews_api").classList.replace("fade-in", "fade-out") 
  : document.querySelector(".cta_previews_api").classList.add("fade-out");

  document.querySelector(".cta_previews_trivia").classList.contains("fade-in") 
  ? document.querySelector(".cta_previews_trivia").classList.replace("fade-in", "fade-out") 
  : document.querySelector(".cta_previews_trivia").classList.add("fade-out");

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

  document.querySelector(".cta_previews_pern").addEventListener("mouseenter", function() {
    addImage('./images/extrevPERN.PNG');
    addDesc(extrevDesc);
    }, false); 
    
  document.querySelector(".cta_previews_pern").addEventListener("mouseleave", function() {
    document.querySelector("#sitePreview").innerHTML = null;
    document.querySelector("#sitePreviewDesc").innerHTML = null;
    }, false);

    document.querySelector(".cta_previews_pern").addEventListener("click", function() {
    window.open('https://pern-extrev.herokuapp.com/', '_blank');
  });

  document.querySelector(".cta_previews_sarpy").addEventListener("mouseenter", function() {
    addImage('./images/sarpy.PNG');
    addDesc(sarpyDesc);
    }, false); 
    
  document.querySelector(".cta_previews_sarpy").addEventListener("mouseleave", function() {
    document.querySelector("#sitePreview").innerHTML = null;
    document.querySelector("#sitePreviewDesc").innerHTML = null;
    }, false);

    document.querySelector(".cta_previews_sarpy").addEventListener("click", function() {
    window.open('https://sarpy-county.vercel.app/', '_blank');
  });

  document.querySelector(".cta_previews_api").addEventListener("mouseenter", function() {
    addImage('./images/cityAPIweather.PNG');
    addDesc(cityDesc);
    }, false); 
    
  document.querySelector(".cta_previews_api").addEventListener("mouseleave", function() {
    document.querySelector("#sitePreview").innerHTML = null;
    document.querySelector("#sitePreviewDesc").innerHTML = null;
    }, false);

    document.querySelector(".cta_previews_api").addEventListener("click", function() {
    window.open('https://city-explorer.vercel.app/', '_blank');
  });

  document.querySelector(".cta_previews_fp").addEventListener("mouseover", function() {
    addImage('./images/fp3.PNG');
    addDesc(frontpageDesc);
    }, false); 

  document.querySelector(".cta_previews_fp").addEventListener("mouseleave", function() {
    document.querySelector("#sitePreview").innerHTML = null;
    document.querySelector("#sitePreviewDesc").innerHTML = null;
    }, false);

    document.querySelector(".cta_previews_fp").addEventListener("click", function() {
    window.open('https://front-page-sigma.vercel.app/', '_blank');
  });

  document.querySelector(".cta_previews_trivia").addEventListener("mouseover", function() {
    addImage('./images/trivia.PNG');
    addDesc(triviaDesc);
    }, false); 

  document.querySelector(".cta_previews_trivia").addEventListener("mouseleave", function() {
    document.querySelector("#sitePreview").innerHTML = null;
    document.querySelector("#sitePreviewDesc").innerHTML = null;
    }, false);

    document.querySelector(".cta_previews_trivia").addEventListener("click", function() {
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
