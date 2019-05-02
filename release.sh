#!/bin/bash

################################################################################
# Copyright 2019 UChicago Argonne, LLC.
# (c.f. AUTHORS, LICENSE)
#
# This file is part of the AML project.
# For more info, see https://xgitlab.cels.anl.gov/argo/aml
#
# SPDX-License-Identifier: BSD-3-Clause
################################################################################

##################################
# usage:
##################################

function usage(){
    echo "./release.sh <ACTION> job_id project_id job_token tag (description)"
    echo ""
    echo "ACTION:          Can be one of CREATE, UPDATE or DELETE. If CREATE and"
    echo "                 release exists, then it is updated. If update and release"
    echo "                 does not exists, then it is created."
    echo "job_id:          the identifier of CI job $CI_JOB_ID which created "
    echo "                 artifacts. Can be found in artifact URL"
    echo "project_id:      The AML project gitlab id $CI_PROJECT_ID. By this time"
    echo "                 it is 1070."
    echo "job_token:       The CI job private token $CI_JOB_TOKEN to access"
    echo "                 release API."
    echo "tag:             The branch or tag of commit $CI_COMMIT_REF_NAME."
    echo "                 It is assumed to be whether "master" or vX.X.X,"
    echo "                 otherwise the script will exit."
}

##################################
# Process Arguments and Set Globals
##################################

ARGC=5

if [ $# -lt $ARGC ]; then
    usage
    echo "Wrong number($#) of arguments provided. $ARGC arguments required" 1>&2
    echo "Command line was:"
    echo "$*"
    exit 1
fi

JOB_ID=$2
PROJECT_ID=$3
JOB_TOKEN=$4
TAG=$5

RELEASE_VERSION="" # set to "latest" or "X.X.X" from $TAG
BASE_URL="https://xgitlab.cels.anl.gov/api/v4/projects"
URL="$BASE_URL/$PROJECT_ID"
TOKEN_HEADER="PRIVATE-TOKEN: $JOB_TOKEN"

# Check release version match format
if [ "$TAG" = "master" ]; then
    RELEASE_VERSION="latest"
else
    echo $TAG | grep -q -E 'v[0-9]+\.[0-9]+\.[0-9]+'
    if [ $? -ne 0 ]; then
	usage
	echo "Argument \"$TAG\" is not \"master\" or in format \"vX.X.X\"" 1>&2
	exit 1
    fi
    RELEASE_VERSION=$(echo $TAG | cut -c 2-)
fi

NAME="AML Release $RELEASE_VERSION"

##################################
# Check Remote Access to Release
##################################

OUT=$(curl -s --header "PRIVATE-TOKEN: $JOB_TOKEN" "$URL")
if [ -z "$OUT" ]; then
    echo "URL:$URL did not return response. Check connection." 1>&2
    exit 1
fi
if [ ! -z "$(echo $OUT | grep '404 Project Not Found')" ]; then
    echo "Project id:$PROJECT_ID is not valid. Please check." 1>&2
    curl --header "PRIVATE-TOKEN: $JOB_TOKEN" "$BASE_URL"
    exit 1
fi
if [ ! -z "$(echo $OUT | grep '401 Unauthorized')" ]; then
    echo "Token:$JOB_TOKEN is not valid. Please check." 1>&2
    exit 1
fi

# Now will work on releases directory only
URL="$URL/releases"

##################################
# Look for Artifacts Names and Format 
##################################

EXTENSIONS=("tar.gz" "zip" "bz2" "bz" "tar")
ARTIFACTS_BASE_URL="https://xgitlab.cels.anl.gov/argo/aml/-/jobs/$JOB_ID/artifacts/file"

#Takes extension name as first argument
function exist_result_ext(){
    if [ ! -d result ]; then
	echo "Expected to find './result' directory containing the result of make dist." 1>&2
        return
    fi

    for f in $(ls result); do
	regex=$(echo .+.$1$)
	if [ ! -z $(echo $f | grep -E $regex) ]; then
	    echo $f
	    return
	fi
    done
}

#Takes extension name as first argument
function artifact_name_from_ext(){
    echo "aml-$RELEASE_VERSION.$1"
}

function artifact_checksum(){
    sha256sum "result/$1" | cut -d " " -f 1 > CHECKSUM
    echo "$ARTIFACTS_BASE_URL/CHECKSUM"
}

#Takes extension name as first argument
function artifact_url(){
    echo "$ARTIFACTS_BASE_URL/result/$1"
}

function artifact_upload_from_ext(){
    ART_FILENAME=$(exist_result_ext $1)
    ART_NAME=$(artifact_name_from_ext $1)
    ART_URL=$(artifact_url $ART_FILENAME)    
    if [ ! -z "$ART_FILENAME" ]; then	
	curl --header "$TOKEN_HEADER" \
	     --data name="$ART_NAME" \
	     --data url="$ART_URL" \
	     --request POST "$URL/$TAG/assets/links"

	CHECKSUM_URL=$(artifact_checksum $ART_FILENAME)
	
	curl --header "$TOKEN_HEADER" \
	     --data name="CHECKSUM" \
	     --data url="$CHECKSUM_URL" \
	     --request POST "$URL/$TAG/assets/links"
    fi
}

function artifacts_get_ids(){
        LINKS=$(curl --header "$TOKEN_HEADER" --request GET "$URL/$TAG/assets/links")
        echo $LINKS | grep -oE "\"id\"\:[0-9]+" | cut -c 6-
}

function artifacts_rm_link(){
    curl --header "$TOKEN_HEADER" \
	 --request DELETE "$URL/$TAG/assets/links/$1"
}

function description_update(){
	    curl -s \
		 --header "$CONTENT_HEADER" \
		 --header "$TOKEN_HEADER" \
		 --data name=$NAME \
		 --data tag_name=$TAG \
		 --data description="$1" \
		 --request PUT $URL/$TAG
}

##################################
# Build curl Commands/Arguments 
##################################

set -x

# Check if releases exists
curl -s --header "PRIVATE-TOKEN: $JOB_TOKEN" $URL/$TAG | grep -q $RELEASE_VERSION
EXISTS_RELEASE=$?

function create_release(){
    if [ $EXISTS_RELEASE -eq 0 ]; then
	update_release $*
    else
	CONTENT_HEADER="Content-Type: application/json"
	if [ ! -z "$6" ]; then
	    DESCRIPTION="$6"
	else
	    DESCRIPTION="This AML $TAG Release"
	fi
	COUNT=0
	ARTIFACTS=""

	for ext in ${EXTENSIONS[@]}; do
	    ART_FILENAME=$(exist_result_ext $1)
	    ART_NAME=$(artifact_name_from_ext $1)
	    ART_URL=$(artifact_url $ART_FILENAME)    
	    if [ ! -z "$ART_FILENAME" ]; then
		CHECKSUM_URL=$(artifact_checksum $ART_FILENAME)
		ARTIFACTS="$ARTIFACTS { \"name\": \"$ART_NAME\", \"url\": \"$ART_URL\" }, { \"name\": \"CHECKSUM\", \"url\": \"$CHECKSUM_URL\" }"
		COUNT=$(($COUNT + 1))
	    fi
	done
	
	if [ $COUNT -gt 0 ]; then
	    ARTIFACTS="{ \"links\": [ ${ARTIFACTS::-2} ] }"
	fi

	DATA="{ \"name\": \"$NAME\", \"tag_name\": \"$TAG\", \"ref\": \"$TAG\", \"assets\": $ARTIFACTS, \"description\": \"$DESCRIPTION\"}"
	curl -s \
	     --header "$CONTENT_HEADER" \
	     --header "$TOKEN_HEADER" \
	     --data "$DATA" \
	     --request POST $URL
    fi
}

# Does not delete existing artifacts but
# replaces pointer to the good artifact.
function update_release(){
    if [ $EXISTS_RELEASE -ne 0 ]; then
	create_release $*
    else
	#Update description
	if [ ! -z "$6" ]; then
	    description_update $6
	fi

	#Delete existing assets
	for ID in $(artifacts_get_ids); do
	    artifacts_rm_link $ID
	done
	
	#Upload new assets
	for ext in ${EXTENSIONS[@]}; do
	    artifact_upload_from_ext $ext
	done
    fi
}

# Does not delete existing artifacts
function delete_release(){
    if [ $EXISTS_RELEASE -ne 0 ]; then
	echo "Release $TAG cannot be found at $URL."
    else
	curl -s \
	     --header "$TOKEN_HEADER" \
	     --request DELETE $URL/$TAG
    fi
}

##################################
# Run Distant Modifications 
##################################

function main(){
    if [ "$1" = "DELETE" ]; then
	delete_release $*
    elif [ "$1" = "CREATE" ]; then
	create_release $*
    elif [ "$1" = "UPDATE" ]; then
	update_release $*
    else
	usage
	echo "Wrong ACTION argument." 1>&2
    fi
}

main $*

