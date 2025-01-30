# Release Note
Copyright 2024-, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
 
This software may be subject to U.S. export control laws. By accepting this software, the user agrees to comply with all applicable U.S. export laws and regulations. User has the responsibility to obtain export licenses, or other export authority as may be required before exporting such information to foreign countries or providing access to foreign persons.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

# river-spatial-scale
Repo for code to estimate river spatial scale parameters (e.g., along-river wse covariance/spectra etc) and apply them in a Bayes reconstruction approach to optimize noise-versus-resolution trade-offs using hte multitemporal stack of information from SWOT.

The hydrochron.py script grabs data for the Ocmulgee, Colorado, and Yellowstone rivers and puts them in a pandas dataframe

The process_river_stretch.py script creates the multitemporal stack of multi-reach stretches an runs Bayes reconstruction
returning RiverSreatchData product instances.


# intial code used in flow wave paper:
Hana R. Thurman, George, H. Allen, Brent A. WIlliams, Arnaud Cerbelaud, and Cedric David "SWOT Captures Hydrologic Waves Traveling Down Rivers," Geophysical Research Letters, 2025 (in Review).

