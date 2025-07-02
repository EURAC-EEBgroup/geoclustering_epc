if(!window.dash_clientside) {window.dash_clientside = {};}

window.dash_clientside.clientside = {

  project_onscroll: function(id) {

    window.onscroll = function() {
        // Skip logic if on small screens
        if (window.innerWidth < 1199) {
            return window.dash_clientside.no_update;
        }

        // Get positions of each card section
        var top_analysis = document.getElementById('prj_card_analysis').offsetTop;
        var top_processing = document.getElementById('prj_card_processing').offsetTop;
        var top_clustering = document.getElementById('prj_card_clustering').offsetTop;
        
        // Get current scroll position with offset
        var windowY = window.scrollY+70;
        
        // Reset all button colors to default
        const defaultColor = "rgb(134, 142, 150)";
        const activeColor = "rgb(121, 80, 242)";
        document.getElementById("btnAnalysis").style.color = defaultColor;
        document.getElementById("btnProcessing").style.color = defaultColor;
        document.getElementById("btnClustering").style.color = defaultColor;

        // Set active button based on scroll position
        // Check which section the current scroll position is within
        if (windowY >= top_analysis && windowY < top_processing) {
            document.getElementById("btnAnalysis").focus();
            document.getElementById("btnAnalysis").style.color = activeColor;
        } else if (windowY >= top_processing && windowY < top_clustering) {
            document.getElementById("btnProcessing").focus();
            document.getElementById("btnProcessing").style.color = activeColor;
        } else if (windowY >= top_clustering) {
            document.getElementById("btnClustering").focus();
            document.getElementById("btnClustering").style.color = activeColor;
        }
    };

    return window.dash_clientside.no_update;
  },

  project_navBtnClick: function(btnID, cardID) {
        // Get target element and calculate position
        var myElement = document.getElementById(cardID);
        var topPos = myElement.offsetTop-60;

        // Scroll to element if button ID exists
        if (btnID) {
            window.scrollTo({top: topPos, behavior: 'smooth'});
        }
        
        console.log("Scrolling to position:", topPos);
        return window.dash_clientside.no_update;
  },
}