document.addEventListener("DOMContentLoaded", function() {  //ES6 version of $(document).ready
  document.getElementById("nav-item-container").style.visibility = "visible";  //This is ES6 = No '#'
  document.querySelector(".site_p_pern").style.visibility = "hidden";
  document.querySelector(".site_p_sarpy").style.visibility = "hidden";
  document.querySelector(".site_p_fp").style.visibility = "hidden";
  document.querySelector(".site_p_api").style.visibility = "hidden";
  //document.querySelector(".cta_previews_api").style.opacity = 0;  Not ready yet

//TO ES6 NOTES ==> .toggleClass is now .classList.toggle // $ is now document.querySelector // Removed .stop(true) but did not replace it - do I need to?

document.querySelector(".cta").addEventListener("click", function() {

  document.querySelector(".site_p_pern").style.visibility = "visible";
  document.querySelector(".site_p_api").style.visibility = "visible";
  document.querySelector(".site_p_sarpy").style.visibility = "visible";
  document.querySelector(".site_p_fp").style.visibility = "visible";
  /*
  document.querySelector(".cta_previews_api").classList.contains("fade-out")
  ? document.querySelector(".cta_previews_api").classList.replace("fade-out", "fade-in") 
  : document.querySelector(".cta_previews_api").classList.add("fade-in");
*/
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

  }, false); 

document.querySelector( ".cta_wipe" ).addEventListener("click", function(e) {

  document.querySelector(".site_p_pern").style.visibility = "hidden";
  document.querySelector(".site_p_sarpy").style.visibility = "hidden";
  document.querySelector(".site_p_fp").style.visibility = "hidden";
  document.querySelector(".site_p_api").style.visibility = "hidden";
  
  /*
  document.querySelector(".cta_previews_api").classList.contains("fade-in") 
  ? document.querySelector(".cta_previews_api").classList.replace("fade-in", "fade-out") 
  : document.querySelector(".cta_previews_api").classList.add("fade-out");
*/
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

  }, false); 

  /*
document.querySelector(".cta_previews_api").addEventListener("mouseover", function() {
  document.querySelector(".site_p_api").classList.contains("fade-out")
  ? document.querySelector(".site_p_api").classList.replace("fade-out", "fade-in") 
  : document.querySelector(".site_p_api").classList.add("fade-in");
}, false); 

document.querySelector(".cta_previews_api").addEventListener("mouseleave", function() {
  document.querySelector(".site_p_api").classList.contains("fade-in") 
  ? document.querySelector(".site_p_api").classList.replace("fade-in", "fade-out") 
  : document.querySelector(".site_p_api").classList.add("fade-out");
  }, false);

  document.querySelector(".cta_previews_api").addEventListener("click", function() {
    window.open('https://front-page-sigma.vercel.app/', '_blank');
  });
*/

  document.querySelector(".cta_previews_pern").addEventListener("mouseenter", function() {
    document.querySelector(".site_p_pern").classList.contains("fade-out") 
    ? document.querySelector(".site_p_pern").classList.replace("fade-out", "fade-in") 
    : document.querySelector(".site_p_pern").classList.add("fade-in");
    }, false); 
    
  document.querySelector(".cta_previews_pern").addEventListener("mouseleave", function() {
    document.querySelector(".site_p_pern").classList.contains("fade-in") 
    ? document.querySelector(".site_p_pern").classList.replace("fade-in", "fade-out") 
    : document.querySelector(".site_p_pern").classList.add("fade-out");
    }, false);

    document.querySelector(".cta_previews_pern").addEventListener("click", function() {
    window.open('https://pern-extrev.herokuapp.com/', '_blank');
  });

  document.querySelector(".cta_previews_sarpy").addEventListener("mouseenter", function() {
    document.querySelector(".site_p_sarpy").classList.contains("fade-out") 
    ? document.querySelector(".site_p_sarpy").classList.replace("fade-out", "fade-in") 
    : document.querySelector(".site_p_sarpy").classList.add("fade-in");
    }, false); 
    
  document.querySelector(".cta_previews_sarpy").addEventListener("mouseleave", function() {
    document.querySelector(".site_p_sarpy").classList.contains("fade-in") 
    ? document.querySelector(".site_p_sarpy").classList.replace("fade-in", "fade-out") 
    : document.querySelector(".site_p_sarpy").classList.add("fade-out");
    }, false);

    document.querySelector(".cta_previews_sarpy").addEventListener("click", function() {
    window.open('https://sarpy-county.vercel.app/', '_blank');
  });

  document.querySelector(".cta_previews_api").addEventListener("mouseenter", function() {
    document.querySelector(".site_p_api").classList.contains("fade-out") 
    ? document.querySelector(".site_p_api").classList.replace("fade-out", "fade-in") 
    : document.querySelector(".site_p_api").classList.add("fade-in");
    }, false); 
    
  document.querySelector(".cta_previews_api").addEventListener("mouseleave", function() {
    document.querySelector(".site_p_api").classList.contains("fade-in") 
    ? document.querySelector(".site_p_api").classList.replace("fade-in", "fade-out") 
    : document.querySelector(".site_p_api").classList.add("fade-out");
    }, false);

    document.querySelector(".cta_previews_api").addEventListener("click", function() {
    window.open('https://city-explorer.vercel.app/', '_blank');
  });

  document.querySelector(".cta_previews_fp").addEventListener("mouseover", function() {
    document.querySelector(".site_p_fp").classList.contains("fade-out") 
    ? document.querySelector(".site_p_fp").classList.replace("fade-out", "fade-in") 
    : document.querySelector(".site_p_fp").classList.add("fade-in");
    }, false); 

  document.querySelector(".cta_previews_fp").addEventListener("mouseleave", function() {
    document.querySelector(".site_p_fp").classList.contains("fade-in") 
    ? document.querySelector(".site_p_fp").classList.replace("fade-in", "fade-out") 
    : document.querySelector(".site_p_fp").classList.add("fade-out");
    }, false);

    document.querySelector(".cta_previews_fp").addEventListener("click", function() {
    window.open('https://front-page-sigma.vercel.app/', '_blank');
  });

}); // End Ready

var buttons = document.querySelectorAll(".toggle-button");
var modal = document.querySelector("#modal");

[].forEach.call(buttons, function (button) {
  button.addEventListener("click", function () {
    modal.classList.toggle("off");
  });
});
